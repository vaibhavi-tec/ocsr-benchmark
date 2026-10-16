"""Load the BioVista OCSR dataset as (image_path, true_smiles) pairs."""
import random
import re
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent

# Image names look like '100_6qm7_image_molminer_bbox_2-0.png':
# the first two '_' fields are the pdf_name, the number after the last '-' is the index.
_FILENAME_RE = re.compile(r"^(\d+_[^_]+)_.*?\d+-(\d+)\.png$")


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)["biovista"]


def parse_filename(name: str):
    """Return (pdf_name, index) from an image filename, or None if it doesn't match."""
    m = _FILENAME_RE.match(name)
    return (m.group(1), int(m.group(2))) if m else None


def load_biovista_ocsr(config_path: Path = ROOT / "config.yaml",
                       sample_n: int | None = None, seed: int = 0, verbose: bool = True):
    """Return a list of (image_path, true_smiles) pairs.

    Images are matched to labels by (pdf_name, index) parsed from the filename,
    never by row order. Images without a label and labels without an image are
    reported (and skipped). If sample_n is given, a fixed random sample of that
    size is returned (reproducible via seed).
    """
    cfg = load_config(config_path)
    base = ROOT / cfg["output_dir"]
    img_dir = base / cfg["images_subdir"]
    labels = pd.read_csv(base / cfg["labels_file"])

    # (pdf_name, index) -> smiles
    label_map = {(r.pdf_name, int(r.index)): r.smiles
                 for r in labels.itertuples(index=False)}

    pairs, unlabeled, unparsed = [], [], []
    seen = set()
    for img in sorted(img_dir.glob("*.png")):
        key = parse_filename(img.name)
        if key is None:
            unparsed.append(img.name)
        elif key in label_map:
            pairs.append((str(img), label_map[key]))
            seen.add(key)
        else:
            unlabeled.append(img.name)
    missing_images = sorted(set(label_map) - seen)

    if verbose:
        print(f"Matched {len(pairs)} image/SMILES pairs")
        print(f"Images without labels: {len(unlabeled) + len(unparsed)}"
              f" (unparseable filenames: {len(unparsed)})")
        print(f"Labels without images: {len(missing_images)}")
        for name in (unlabeled + unparsed)[:5]:
            print(f"  no label: {name}")
        for key in missing_images[:5]:
            print(f"  no image: {key}")

    if sample_n is not None and sample_n < len(pairs):
        pairs = random.Random(seed).sample(pairs, sample_n)
    return pairs
