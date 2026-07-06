"""
setup_dataset.py — Extract the YOLO dataset zip into datasets/new_final_yolo/

Usage (run once, from the project root):
    python setup_dataset.py
    python setup_dataset.py --zip "path/to/YOLO MODEl.zip"

The script is idempotent — it skips files that already exist.
"""
import argparse
import shutil
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DEST = PROJECT_ROOT / "datasets" / "new_final_yolo"

# Candidate zip locations (checked in order)
_CANDIDATE_ZIPS = [
    PROJECT_ROOT / "datasets" / "YOLO_MODEl.zip",
    PROJECT_ROOT / "datasets" / "YOLO MODEl.zip",
    PROJECT_ROOT / "YOLO_MODEl.zip",
    PROJECT_ROOT / "YOLO MODEl.zip",
]


def find_zip() -> Path | None:
    for p in _CANDIDATE_ZIPS:
        if p.exists():
            return p
    return None


def extract(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.infolist() if not m.filename.endswith("/")]
        total = len(members)
        print(f"  Archive contains {total} files → {dest}")

        done = skipped = 0
        for i, member in enumerate(members):
            out = dest / member.filename
            if out.exists():
                skipped += 1
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
            done += 1
            if done % 500 == 0:
                pct = (done + skipped) / total * 100
                print(f"    {done + skipped}/{total} ({pct:.0f}%)  extracted={done}  skipped={skipped}")

    print(f"\n  Done — {done} extracted, {skipped} already present")
    _verify(dest)


def _verify(dest: Path) -> None:
    train = len(list((dest / "train" / "images").glob("*.jpg"))) if (dest / "train" / "images").exists() else 0
    valid = len(list((dest / "valid" / "images").glob("*.jpg"))) if (dest / "valid" / "images").exists() else 0
    test  = len(list((dest / "test"  / "images").glob("*.jpg"))) if (dest / "test"  / "images").exists() else 0
    yaml_ok = (dest / "data.yaml").exists()
    print(f"\n  Verification:")
    print(f"    train images : {train:,}  (expected 7,586)")
    print(f"    valid images : {valid:,}  (expected 948)")
    print(f"    test  images : {test:,}  (expected 948)")
    print(f"    data.yaml    : {'✓' if yaml_ok else '✗ MISSING'}")
    if train < 7000 or valid < 900 or not yaml_ok:
        print("\n  WARNING: extraction may be incomplete — re-run this script.")
    else:
        print("\n  Dataset ready for training.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", default=None, help="Path to the YOLO MODEl.zip file")
    ap.add_argument("--dest", default=str(DEFAULT_DEST), help="Destination directory")
    args = ap.parse_args()

    zip_path = Path(args.zip) if args.zip else find_zip()
    if zip_path is None:
        print("ERROR: Could not find YOLO_MODEl.zip. Place it in the project root or datasets/ folder,")
        print("       or pass --zip <path>")
        sys.exit(1)

    if not zip_path.exists():
        print(f"ERROR: zip not found: {zip_path}")
        sys.exit(1)

    dest = Path(args.dest)
    print(f"Extracting: {zip_path.name}  ({zip_path.stat().st_size / 1e9:.2f} GB)")
    print(f"       To : {dest}\n")
    extract(zip_path, dest)


if __name__ == "__main__":
    main()
