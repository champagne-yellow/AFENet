# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import shutil
import subprocess
from hashlib import sha256
from pathlib import Path

import torch

BLOCK_SIZE = 128 * 1024


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process a checkpoint to be published'
    )
    parser.add_argument('in_file', help='input checkpoint filename')
    parser.add_argument('out_file', help='output checkpoint filename')
    return parser.parse_args()


def compute_sha256_hash(filename: str) -> str:
    """Compute SHA256 hash digest from a file.

    Args:
        filename: Path to the file to hash.

    Returns:
        SHA256 hash digest as hexadecimal string.
    """
    hash_func = sha256()
    byte_array = bytearray(BLOCK_SIZE)
    memory_view = memoryview(byte_array)

    with open(filename, 'rb', buffering=0) as file:
        for block in iter(lambda: file.readinto(memory_view), 0):
            hash_func.update(memory_view[:block])

    return hash_func.hexdigest()


def process_checkpoint(in_file: str, out_file: str):
    """Process checkpoint by removing optimizer and adding hash suffix.

    Args:
        in_file: Input checkpoint file path.
        out_file: Output checkpoint file path.
    """
    # Load checkpoint
    checkpoint = torch.load(in_file, map_location='cpu')

    # Remove optimizer for smaller file size
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']

    # Add code here to remove sensitive data from checkpoint['meta'] if needed

    # Save processed checkpoint
    torch.save(checkpoint, out_file)

    # Compute hash and rename file
    file_hash = compute_sha256_hash(in_file)
    final_filename = Path(out_file).stem + f'-{file_hash[:8]}.pth'
    final_path = Path(out_file).parent / final_filename

    # Use shutil for cross-platform file moving
    shutil.move(out_file, final_path)

    print(f"Processed checkpoint saved to: {final_path}")
    print(f"SHA256 hash (first 8 chars): {file_hash[:8]}")


def main():
    """Main function to process checkpoint."""
    args = parse_args()

    # Validate input file exists
    if not Path(args.in_file).exists():
        raise FileNotFoundError(f"Input file not found: {args.in_file}")

    # Validate output directory exists
    output_dir = Path(args.out_file).parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    process_checkpoint(args.in_file, args.out_file)


if __name__ == '__main__':
    main()