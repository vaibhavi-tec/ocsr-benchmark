"""MolGlyph wrapper (BioMiner's open-weight OCSR model, Yan et al., 2026).

MolGlyph outputs extended SMILES (E-SMILES): a SMILES string, then '<sep>', then
annotations such as <a>0:SEM</a> (atom 0 is the abbreviation SEM). Before scoring,
each output is converted to plain SMILES exactly as BioMiner does in
BioMiner/commons/ocsr.py:  get_refactor(Translator.refactor(raw_output))
"""
import importlib
from typing import List

from .base import OCSRTool
from .vendored import download_checkpoint, import_molscribe, pick_device


class MolGlyphTool(OCSRTool):
    name = "molglyph"

    def __init__(self, hf_repo: str, hf_file: str, models_dir: str,
                 device: str = "auto", batch_size: int = 16):
        super().__init__(batch_size)
        ckpt = download_checkpoint(hf_repo, hf_file, models_dir)  # gated: needs HF_TOKEN
        self.device = pick_device(device)
        package = import_molscribe()
        self.model = package.MolGlyph(ckpt, self.device)
        # BioMiner's E-SMILES converter (vendored next to the model code)
        self.utils = importlib.import_module(package.__name__ + ".molglyph_utils")

    def _to_plain_smiles(self, raw):
        """Apply BioMiner's post-processing; '' or None means conversion failed."""
        try:
            return self.utils.get_refactor(self.utils.Translator.refactor(raw)) or None
        except Exception:
            return None  # scored as an invalid prediction, not a crash

    def _predict_batch(self, image_paths: List[str]) -> List[dict]:
        # MolGlyph returns plain strings (MolScribe returns dicts) and no confidence score
        outputs = self.model.predict_image_files(image_paths, return_atoms_bonds=False,
                                                 return_confidence=False)
        results = []
        for out in outputs:
            raw = out.get("smiles") if isinstance(out, dict) else out
            results.append({"smiles": self._to_plain_smiles(raw), "confidence": None})
        return results