# biominer_molscribe

Vendored copy of the `BioMiner/MolScribe/molscribe/` package, which contains both
the `MolScribe` and `MolGlyph` OCSR models.

- Source repo: https://github.com/jiaxianyan/BioMiner
- Commit: `17c6161b40e34e2c377b2871f760baab3fdeea06`
- Source path: `BioMiner/MolScribe/molscribe/`
- Licenses: `LICENSE` (BioMiner, MIT, Copyright (c) 2025 jiaxianyan) and
  `LICENSE.MolScribe` (original MolScribe, MIT, Copyright (c) 2022 Yujie Qian)

## Changes
The Python code is unmodified. The only difference from upstream is that the
Linux-only native Indigo binaries (`indigo/lib/`) were not copied. Indigo is only
loaded when `Indigo()` is instantiated (training-data augmentation in `dataset.py`),
not for inference.

## Added files

- `molglyph_utils.py` and `abbrevs.csv` copied unmodified from `BioMiner/commons/` at the same commit. They convert MolGlyph's extended SMILES (E-SMILES) output into plain SMILES, as BioMiner does in `BioMiner/commons/ocsr.py`.
