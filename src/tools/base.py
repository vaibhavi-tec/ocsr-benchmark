"""Common interface that every OCSR tool wrapper implements.

A tool takes a list of image paths and returns one Prediction per image, in the
same order. Batches that fail are retried one image at a time, so a single bad
image records an error instead of stopping the whole benchmark.
"""
import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Prediction:
    image_path: str
    pred_smiles: Optional[str]     # None if the tool failed on this image
    confidence: Optional[float]    # None if the tool doesn't report one
    runtime_s: float               # seconds spent on this image (batch time / batch size)
    error: Optional[str] = None    # error message if the tool failed


class OCSRTool:
    """Base class. Subclasses implement _predict_batch()."""

    name = "base"

    def __init__(self, batch_size: int = 16):
        self.batch_size = batch_size

    def _predict_batch(self, image_paths: List[str]) -> List[dict]:
        """Return one dict per image with keys 'smiles' and optionally 'confidence'."""
        raise NotImplementedError

    def predict(self, image_paths: List[str]) -> List[Prediction]:
        results: List[Prediction] = []
        for start in range(0, len(image_paths), self.batch_size):
            batch = image_paths[start:start + self.batch_size]
            results.extend(self._run(batch))
        return results

    def _run(self, batch: List[str]) -> List[Prediction]:
        t0 = time.perf_counter()
        try:
            outputs = self._predict_batch(batch)
            per_image = (time.perf_counter() - t0) / len(batch)
            return [Prediction(path, out.get("smiles"), _to_float(out.get("confidence")), per_image)
                    for path, out in zip(batch, outputs)]
        except Exception as exc:
            if len(batch) == 1:
                return [Prediction(batch[0], None, None, time.perf_counter() - t0,
                                   f"{type(exc).__name__}: {exc}")]
            # Retry one by one to find which image(s) caused the failure
            return [p for path in batch for p in self._run([path])]


def _to_float(value) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
