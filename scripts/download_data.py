"""Download and unpack the BioVista OCSR subset from Google Drive.

Settings (file ID, output folder) come from config.yaml, nothing is hardcoded.
Usage: python scripts/download_data.py [--config config.yaml]
"""
import argparse
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

import gdown
import yaml

# Project root = parent of the scripts/ folder, so paths work from any cwd
ROOT = Path(__file__).resolve().parent.parent

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"}


def load_config(path: Path) -> dict:
    """Read the YAML config and return its 'biovista' section."""
    with open(path) as f:
        return yaml.safe_load(f)["biovista"]


def data_exists(out_dir: Path) -> bool:
    """The data counts as present if the output folder exists and is non-empty."""
    return out_dir.is_dir() and any(out_dir.iterdir())


def download_and_extract(file_id: str, out_dir: Path) -> None:
    """Fetch the file from Google Drive; extract it if it is a zip or tar archive."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Download into a temp dir so a failed download leaves no partial data behind
    with tempfile.TemporaryDirectory() as tmp:
        # output ending in '/' makes gdown keep the original filename from Drive
        downloaded = gdown.download(id=file_id, output=tmp + "/", quiet=False)
        if downloaded is None:
            raise RuntimeError("gdown failed (Drive quota or permission problem?)")
        downloaded = Path(downloaded)
        if zipfile.is_zipfile(downloaded):
            with zipfile.ZipFile(downloaded) as z:
                z.extractall(out_dir)
        elif tarfile.is_tarfile(downloaded):
            # BioVista's OCSR file is a .tar.gz
            with tarfile.open(downloaded) as t:
                t.extractall(out_dir, filter="data")
        else:
            # Not an archive: just keep the file as-is
            shutil.move(str(downloaded), out_dir / downloaded.name)


def summarize(out_dir: Path) -> None:
    """Print total size, image count, image formats and CSV files."""
    files = [p for p in out_dir.rglob("*") if p.is_file()]
    images = [p for p in files if p.suffix.lower() in IMAGE_EXTS]
    csvs = [p for p in files if p.suffix.lower() == ".csv"]
    total_mb = sum(p.stat().st_size for p in files) / 1e6
    print(f"\nSummary for {out_dir}")
    print(f"  total size : {total_mb:.1f} MB ({len(files)} files)")
    print(f"  images     : {len(images)} ({sorted({p.suffix.lower() for p in images})})")
    for c in csvs:
        print(f"  csv        : {c.relative_to(out_dir)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    out_dir = ROOT / cfg["output_dir"]

    # Skip the download if we already have the data
    if data_exists(out_dir):
        print(f"Data already exists at {out_dir}, skipping download.")
    else:
        download_and_extract(cfg["gdrive_file_id"], out_dir)

    summarize(out_dir)


if __name__ == "__main__":
    main()
