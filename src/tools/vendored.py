"""Shared loading code for the two tools that run on BioMiner's vendored MolScribe package."""
import os
import sys
from pathlib import Path

# The MolScribe/MolGlyph checkpoints store training arguments alongside the weights.
# torch>=2.6 refuses to unpickle those by default, so allow full loading. Only do
# this for checkpoints from trusted sources (here: the official Hugging Face repos).
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import torch  # noqa: E402  (must come after the environment variable above)
from dotenv import load_dotenv  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
VENDOR_DIR = ROOT / "third_party" / "biominer_molscribe"


def import_molscribe():
    """Import the vendored package (exports MolScribe and MolGlyph).

    Supports both layouts: third_party/biominer_molscribe/molscribe/ (package in
    a subfolder) or third_party/biominer_molscribe/ (package files directly).
    """
    import importlib
    if (VENDOR_DIR / "molscribe" / "__init__.py").exists():
        path, name = VENDOR_DIR, "molscribe"
    elif (VENDOR_DIR / "__init__.py").exists():
        path, name = VENDOR_DIR.parent, VENDOR_DIR.name
    else:
        raise FileNotFoundError(
            f"Vendored code not found in {VENDOR_DIR}. Run: bash scripts/vendor_molscribe.sh")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return importlib.import_module(name)


def pick_device(name: str = "auto") -> torch.device:
    """'auto' -> CUDA if available, else CPU. MPS must be requested explicitly."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def download_checkpoint(repo_id: str, filename: str, models_dir: str) -> str:
    """Download a checkpoint from Hugging Face into models_dir (skipped if present).

    The token is read from the HF_TOKEN environment variable (a local .env file
    or Colab Secrets). It is never printed or written to disk by this code.
    """
    load_dotenv(ROOT / ".env")  # no-op if .env doesn't exist (e.g. on Colab)
    local_dir = ROOT / models_dir
    local_dir.mkdir(parents=True, exist_ok=True)
    return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(local_dir),
                           token=os.getenv("HF_TOKEN") or None)
