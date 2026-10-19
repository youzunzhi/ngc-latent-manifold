"""Download and verify the paper checkpoints from the pinned HF revision."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--include-replicates', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    inventory = json.loads(
        (repo / 'artifacts/checkpoint-inventory.json').read_text()
    )
    source = inventory['huggingface']
    selected = [
        f
        for f in inventory['files']
        if args.include_replicates or f['category'] == 'main_checkpoint'
    ]
    for entry in selected:
        path = Path(
            hf_hub_download(
                repo_id=source['repo_id'],
                revision=source['revision'],
                filename=entry['release_path'],
                local_dir=args.root,
            )
        )
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if (
            path.stat().st_size != entry['size_bytes']
            or digest.hexdigest() != entry['sha256']
        ):
            raise RuntimeError(f'Checkpoint verification failed: {path}')
        print(f'Verified {entry["release_path"]}')


if __name__ == '__main__':
    main()
