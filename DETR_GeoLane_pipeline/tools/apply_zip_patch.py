import argparse
import os
import shutil
import tempfile
import zipfile
from pathlib import Path


def apply_zip(project_root: Path, zip_path: Path):
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(td)
        temp_root = Path(td)
        candidates = [p for p in temp_root.iterdir() if p.is_dir()]
        src_root = candidates[0] if len(candidates) == 1 else temp_root
        for src in src_root.rglob('*'):
            if src.is_dir():
                continue
            rel = src.relative_to(src_root)
            dst = project_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f'[updated] {rel}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', required=True, help='Target project root')
    ap.add_argument('--zip', dest='zip_file', help='Single zip file to apply')
    ap.add_argument('--zip-dir', help='Apply all zip files in a folder, sorted by name')
    args = ap.parse_args()
    project_root = Path(args.project)
    if not project_root.exists():
        raise FileNotFoundError(project_root)
    if args.zip_file:
        apply_zip(project_root, Path(args.zip_file))
    elif args.zip_dir:
        zips = sorted(Path(args.zip_dir).glob('*.zip'))
        if not zips:
            raise FileNotFoundError('No zip files found in zip-dir')
        for zp in zips:
            print(f'Applying {zp.name}')
            apply_zip(project_root, zp)
    else:
        raise ValueError('Provide --zip or --zip-dir')


if __name__ == '__main__':
    main()
