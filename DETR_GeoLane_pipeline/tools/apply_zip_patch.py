
import argparse, os, zipfile, tempfile, shutil
from pathlib import Path

def apply_zip(project: Path, zip_path: Path):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(td)
        entries = [p for p in td.iterdir()]
        src_root = entries[0] if len(entries) == 1 and entries[0].is_dir() else td
        for p in src_root.rglob("*"):
            if p.is_dir():
                continue
            rel = p.relative_to(src_root)
            dst = project / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            print(f"updated: {rel}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--zip", required=True)
    args = ap.parse_args()
    apply_zip(Path(args.project), Path(args.zip))
