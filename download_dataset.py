#!/usr/bin/env python3
"""Download the Kaggle images dataset.

Requires:
    pip install kaggle

Setup:
    1. Go to https://www.kaggle.com/settings → API → Create New Token
    2. Place the downloaded kaggle.json in ~/.kaggle/kaggle.json
    3. chmod 600 ~/.kaggle/kaggle.json
"""

import subprocess
import sys

DATASET = "muratkokludataset/grapevine-leaves-image-dataset"
DEST = "images-dataset"

cmd = ["kaggle", "datasets", "download", "-d", DATASET, "--unzip", "-p", DEST]

print(f"Downloading {DATASET} → ./{DEST}/")
try:
    subprocess.run(cmd, check=True)
except FileNotFoundError:
    sys.exit("Error: 'kaggle' CLI not found. Install it with: pip install kaggle")
except subprocess.CalledProcessError as e:
    sys.exit(f"Error: download failed (exit code {e.returncode}). Check your Kaggle API credentials.")

print("Done.")
