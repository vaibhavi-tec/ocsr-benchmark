"""MolScribe wrapper (image-to-graph OCSR, Qian et al., 2023)."""
from typing import List

from .base import OCSRTool
from .vendored import download_checkpoint, import_molscribe, pick_device


class MolScribeTool(OCSRTool):
    name = "molscribe"

    def __init__(self, hf_repo: str, hf_file: str, models_dir: str,
                 device: str = "auto", batch_size: int = 16):
        super().__init__(batch_size)
        ckpt = download_checkpoint(hf_repo, hf_file, models_dir)
        self.device = pick_device(device)
        self.model = import_molscribe().MolScribe(ckpt, self.device)

    def _predict_batch(self, image_paths: List[str]) -> List[dict]:
        # return_confidence=True adds a 'confidence' score per prediction
        return self.model.predict_image_files(image_paths, return_atoms_bonds=False,
                                              return_confidence=True)
