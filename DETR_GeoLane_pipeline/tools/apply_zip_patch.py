import argparse, os, zipfile, shutil, tempfile
from pathlib import Path

def apply_one(project_dir: Path, zip_path: Path):
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(td)
        td = Path(td)
        roots = [p for p in td.iterdir()]
        source_root = td
        if len(roots) == 1 and roots[0].is_dir() and (roots[0] / 'src').exists():
            source_root = roots[0]
        for src in source_root.rglob('*'):
            if src.is_dir():
                continue
            rel = src.relative_to(source_root)
            dst = project_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f'updated: {rel}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', required=True)
    ap.add_argument('--zip')
    ap.add_argument('--zip-dir')
    args = ap.parse_args()
    project_dir = Path(args.project)
    assert project_dir.exists(), f'project dir not found: {project_dir}'
    zips = []
    if args.zip:
        zips.append(Path(args.zip))
    if args.zip_dir:
        zips.extend(sorted(Path(args.zip_dir).glob('*.zip')))
    assert zips, 'provide --zip or --zip-dir'
    for zp in zips:
        print(f'\nApplying {zp}')
        apply_one(project_dir, zp)
    print('\nDone.')

if __name__ == '__main__':
    main()
