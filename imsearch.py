#!/usr/bin/env python3
"""
imsearch — Image similarity search using Milvus and Qwen3-VL-Embedding-8B.

Requirements:
    pip install torch transformers>=4.57.0 qwen-vl-utils>=0.0.14 pymilvus Pillow

Usage:
    python imsearch.py insert --dir ./photos --tags "nature,outdoor"
    python imsearch.py search --query ./photo.jpg --top-k 10
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pymilvus import DataType, MilvusClient
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLModel,
    Qwen3VLPreTrainedModel,
)

# ── Constants ──────────────────────────────────────────────────────────────

IMAGE_FACTOR = 32
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR  # 4 096
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR  # 1 843 200
MAX_LENGTH = 8192
INSTRUCTION = "Represent the user's input."

COLLECTION = "image_embeddings"
MILVUS_URI = "http://localhost:19530"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ── Qwen3-VL embedding model (no LM head) ─────────────────────────────────


@dataclass
class EmbeddingOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.Tensor] = None


class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
    """Qwen3-VL base transformer wrapped for embedding extraction."""

    def __init__(self, config: Qwen3VLConfig):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        cache_position=None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        return EmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )


# ── Embedder wrapper ──────────────────────────────────────────────────────


class ImageEmbedder:
    """Loads Qwen3-VL-Embedding and produces L2-normalised image embeddings."""

    def __init__(self, model_name: str = "Qwen/Qwen3-VL-Embedding-8B"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32

        print(f"Loading model {model_name} on {device} ({dtype}) ...")
        self.device = device
        self.model = (
            Qwen3VLForEmbedding.from_pretrained(
                model_name, trust_remote_code=True, torch_dtype=dtype
            )
            .to(device)
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name, padding_side="right"
        )

        # Dynamically infer the embedding dimension with a test forward pass.
        self.dim = self._infer_dim()
        print(f"Embedding dimension: {self.dim}")

    def _infer_dim(self) -> int:
        dummy = Image.new("RGB", (128, 128), color=(128, 128, 128))
        return self._embed(dummy).shape[0]

    def _make_messages(self, image):
        """Build chat messages for a single image (file path or PIL Image)."""
        if isinstance(image, str):
            img_ref = f"file://{os.path.abspath(image)}"
        else:
            img_ref = image  # PIL Image passed directly

        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": INSTRUCTION}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_ref,
                        "min_pixels": MIN_PIXELS,
                        "max_pixels": MAX_PIXELS,
                    }
                ],
            },
        ]

    @torch.no_grad()
    def _embed(self, image) -> np.ndarray:
        messages = self._make_messages(image)

        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            do_resize=False,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        out = self.model(**inputs)
        hidden = out.last_hidden_state  # (1, seq_len, hidden_dim)
        mask = out.attention_mask  # (1, seq_len)

        # Last-token pooling: extract the hidden state at the final
        # non-padding position (the EOS / generation-prompt token).
        last_pos = mask.flip(1).argmax(dim=1)
        col = mask.shape[1] - last_pos - 1
        row = torch.arange(hidden.shape[0], device=self.device)
        emb = hidden[row, col]

        emb = F.normalize(emb, p=2, dim=-1)
        return emb[0].cpu().float().numpy()

    def embed_image(self, path: str) -> np.ndarray:
        """Return an L2-normalised embedding vector for an image file."""
        return self._embed(path)


# ── Milvus helpers ─────────────────────────────────────────────────────────


def get_client() -> MilvusClient:
    try:
        return MilvusClient(uri=MILVUS_URI)
    except Exception as exc:
        print(
            f"Error: cannot connect to Milvus at {MILVUS_URI}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


def ensure_collection(client: MilvusClient, dim: int):
    """Create the collection (with COSINE index) if it does not exist."""
    if client.has_collection(COLLECTION):
        return

    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name="file_path", datatype=DataType.VARCHAR, max_length=2048)
    schema.add_field(field_name="file_name", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="width", datatype=DataType.INT64)
    schema.add_field(field_name="height", datatype=DataType.INT64)
    schema.add_field(field_name="tags", datatype=DataType.VARCHAR, max_length=1024)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 256},
    )

    client.create_collection(
        collection_name=COLLECTION, schema=schema, index_params=index_params
    )
    print(f"Created Milvus collection '{COLLECTION}' (dim={dim})")


# ── CLI commands ───────────────────────────────────────────────────────────


def find_images(directory: str) -> list[str]:
    """Recursively find image files under *directory*."""
    paths = []
    for root, _, files in os.walk(directory):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                paths.append(os.path.join(root, fname))
    paths.sort()
    return paths


def cmd_insert(args):
    if not os.path.isdir(args.dir):
        print(f"Error: '{args.dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    image_paths = find_images(args.dir)
    if not image_paths:
        print(f"No images found in '{args.dir}'.")
        return

    total = len(image_paths)
    print(f"Found {total} image(s) in '{args.dir}'")

    embedder = ImageEmbedder()
    client = get_client()
    ensure_collection(client, embedder.dim)

    tags = args.tags or ""
    inserted = 0

    for i, path in enumerate(image_paths, 1):
        name = os.path.basename(path)
        print(f"[{i}/{total}] Inserting {name} ...")

        # Read image to get dimensions (and verify it's readable).
        try:
            img = Image.open(path)
            img.load()
            w, h = img.size
        except Exception as exc:
            print(f"  Warning: cannot read '{path}': {exc}  — skipping")
            continue

        # Generate embedding.
        try:
            emb = embedder.embed_image(path)
        except Exception as exc:
            print(f"  Warning: embedding failed for '{path}': {exc}  — skipping")
            continue

        # Insert into Milvus.
        client.insert(
            collection_name=COLLECTION,
            data=[
                {
                    "embedding": emb.tolist(),
                    "file_path": os.path.abspath(path),
                    "file_name": name,
                    "width": w,
                    "height": h,
                    "tags": tags,
                }
            ],
        )
        inserted += 1

    print(f"\nDone. Inserted {inserted}/{total} image(s) into '{COLLECTION}'.")


def cmd_search(args):
    if not os.path.isfile(args.query):
        print(f"Error: '{args.query}' is not a file.", file=sys.stderr)
        sys.exit(1)

    embedder = ImageEmbedder()
    client = get_client()

    if not client.has_collection(COLLECTION):
        print(
            f"Error: collection '{COLLECTION}' does not exist. "
            "Run 'insert' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Embedding query image: {args.query}")
    emb = embedder.embed_image(args.query)

    results = client.search(
        collection_name=COLLECTION,
        data=[emb.tolist()],
        limit=args.top_k,
        search_params={"metric_type": "COSINE", "params": {"ef": 128}},
        output_fields=["file_path", "file_name", "width", "height", "tags"],
    )

    print(f"\nTop-{args.top_k} results:\n")
    header = f"{'Rank':<6}{'Score':<10}{'Dimensions':<14}{'Tags':<20}{'File'}"
    print(header)
    print("-" * max(len(header), 80))

    for hits in results:
        for rank, hit in enumerate(hits, 1):
            e = hit["entity"]
            dims = f"{e['width']}x{e['height']}"
            print(
                f"{rank:<6}{hit['distance']:<10.4f}{dims:<14}"
                f"{e['tags']:<20}{e['file_path']}"
            )


# ── Entrypoint ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Image similarity search with Milvus & Qwen3-VL-Embedding-8B"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ins = sub.add_parser("insert", help="Index images from a directory")
    p_ins.add_argument("--dir", required=True, help="Directory of images to index")
    p_ins.add_argument(
        "--tags", default="", help='Comma-separated tags, e.g. "nature,outdoor"'
    )

    p_srch = sub.add_parser("search", help="Search for similar images")
    p_srch.add_argument("--query", required=True, help="Path to query image")
    p_srch.add_argument(
        "--top-k", type=int, default=5, help="Number of results (default: 5)"
    )

    args = parser.parse_args()
    {"insert": cmd_insert, "search": cmd_search}[args.command](args)


if __name__ == "__main__":
    main()
