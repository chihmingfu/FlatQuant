#!/usr/bin/env python3
"""
Download C4 dataset for FlatQuant
This script downloads the C4 validation dataset that FlatQuant expects.
"""

import os
import gzip
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
from tqdm import tqdm

def download_c4_validation():
    # Create the expected directory structure
    base_dir = Path("./datasets/allenai/c4/en")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Download validation file
    print("Downloading C4 validation dataset...")
    
    # The file FlatQuant expects
    target_file = "c4-validation.00000-of-00008.json.gz"
    
    try:
        # Download from Hugging Face Hub
        downloaded_file = hf_hub_download(
            repo_id="allenai/c4",
            filename=f"en/{target_file}",
            repo_type="dataset",
            cache_dir="./datasets/.cache"
        )
        
        # Create a symlink or copy to the expected location
        target_path = base_dir / target_file
        if not target_path.exists():
            # Use absolute paths for symlink
            os.symlink(os.path.abspath(downloaded_file), os.path.abspath(target_path))
            print(f"Created symlink: {target_path} -> {downloaded_file}")
        else:
            print(f"File already exists: {target_path}")
            
        # Verify the file
        print("\nVerifying downloaded file...")
        with gzip.open(target_path, 'rt') as f:
            count = 0
            for line in f:
                count += 1
                if count <= 3:
                    data = json.loads(line)
                    print(f"Example {count}: {data['text'][:100]}...")
                    
        print(f"\nTotal examples in validation file: {count}")
        
    except Exception as e:
        print(f"Error downloading C4: {e}")
        print("\nAlternative: You can manually download from:")
        print("https://huggingface.co/datasets/allenai/c4/tree/main/en")
        print(f"And place the file in: {base_dir}")

def download_c4_train_sample():
    """Download a sample of C4 train data (first file only)"""
    base_dir = Path("./datasets/allenai/c4/en")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nDownloading C4 train sample...")
    target_file = "c4-train.00000-of-01024.json.gz"
    
    try:
        downloaded_file = hf_hub_download(
            repo_id="allenai/c4",
            filename=f"en/{target_file}",
            repo_type="dataset",
            cache_dir="./datasets/.cache"
        )
        
        target_path = base_dir / target_file
        if not target_path.exists():
            os.symlink(downloaded_file, target_path)
            print(f"Created symlink: {target_path} -> {downloaded_file}")
        else:
            print(f"File already exists: {target_path}")
            
    except Exception as e:
        print(f"Error downloading C4 train: {e}")

if __name__ == "__main__":
    import sys
    print("Setting up C4 dataset for FlatQuant...")
    
    # Download validation dataset (required)
    download_c4_validation()
    
    # Check if train argument is provided
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        download_c4_train_sample()
    else:
        print("\nTo also download C4 train sample, run: python download_c4.py --train")
    
    print("\nDone! C4 dataset is ready for FlatQuant.")