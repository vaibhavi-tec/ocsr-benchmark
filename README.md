# OCSR Benchmark: MolGlyph (BioMiner) vs. MolScribe on BioVista

This repository benchmarks two optical chemical structure recognition (OCSR) tools, which convert images of molecules into SMILES strings, on the public **BioVista** OCSR dataset (8,861 molecule images from 500 papers, each with a ground-truth SMILES).

- **MolGlyph**: the open-weight OCSR model used inside BioMiner (Yan et al., 2025; 2026)
- **MolScribe**: an image-to-graph OCSR model (Qian et al., 2023), the baseline used in the BioMiner paper

The pipeline runs end to end with one command: **data loading → inference → metrics → output**. On the full dataset it reproduces the published BioVista scores to within 0.001 for MolGlyph and 0.006 for MolScribe (see [Results](#results)).

Course: AAI 6630 Computer Vision, Northeastern University (XN project, Phase 1). Author: Vaibhavi Bogaraja Raj.

## Repository structure

```
ocsr-benchmark/
├── README.md                   this file
├── requirements.txt            pinned dependencies (Python 3.10)
├── config.yaml                 all settings and paths (no machine-specific paths)
├── .env.example                template for the Hugging Face token (real .env is not committed)
├── scripts/
│   ├── download_data.py        downloads and unzips the BioVista OCSR subset
│   └── vendor_molscribe.sh     copies BioMiner's MolScribe/MolGlyph code into third_party/
├── src/
│   ├── data.py                 loads (image, true SMILES) pairs, matched by filename
│   ├── metrics.py              exact match, Tanimoto, validity, official BioVista matcher
│   ├── run_benchmark.py        main script: runs the whole pipeline
│   └── tools/
│       ├── base.py             common interface every tool implements
│       ├── vendored.py         shared model loading (device, checkpoint download)
│       ├── molscribe_tool.py   MolScribe wrapper
│       └── molglyph_tool.py    MolGlyph wrapper (includes E-SMILES conversion)
├── tests/test_metrics.py       checks the metrics on known examples (28 checks)
├── notebooks/run_on_colab.ipynb  runs the full benchmark on a free Colab GPU
├── third_party/biominer_molscribe/  vendored BioMiner code (MIT license, unmodified)
└── results/                    benchmark outputs (CSV and JSON) from the full run
```

`data/` (the dataset), `models/` (checkpoints), `.venv/` and `.env` are excluded from Git.

## Setup

Requirements: Python 3.10 (managed with [uv](https://docs.astral.sh/uv/)), Git, and a free Hugging Face account.

```bash
git clone https://github.com/vaibhavi-tec/ocsr-benchmark.git
cd ocsr-benchmark

# 1. Python 3.10 environment and dependencies
uv venv --python 3.10 .venv          # install uv first: brew install uv  (or pip install uv)
source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Hugging Face token (MolGlyph is a gated model)
#    a) accept the terms at https://huggingface.co/jiaxianustc/MolGlyph
#    b) create a Read token at https://huggingface.co/settings/tokens
cp .env.example .env                 # then paste your token into .env as HF_TOKEN=...

# 3. Dataset (downloads into data/biovista; skipped if already present)
python scripts/download_data.py

# 4. Check the metrics (should print 28/28 checks passed)
python tests/test_metrics.py
```

The vendored BioMiner code is already committed in `third_party/`, so no extra step is needed. Model checkpoints (MolScribe about 1.1 GB, MolGlyph about 1.7 GB) download automatically into `models/` on first use.

## Running the benchmark

```bash
python src/run_benchmark.py --tools molscribe --limit 5    # quick check (works on CPU)
python src/run_benchmark.py --limit 500 --seed 42          # reproducible 500-image sample
python src/run_benchmark.py                                # full benchmark, both tools
```

| Option | Meaning |
|---|---|
| `--tools` | Which tools to run (default: `molscribe molglyph`) |
| `--limit N` | Random sample of N images (default: all, from `config.yaml`) |
| `--seed S` | Sampling seed, so the same sample can be reproduced |
| `--fresh` | Ignore saved progress and rerun every image |

### Running on Google Colab (recommended for the full run)

A full run needs a GPU. Open the notebook directly in Colab:

https://colab.research.google.com/github/vaibhavi-tec/ocsr-benchmark/blob/main/notebooks/run_on_colab.ipynb

Then set **Runtime → Change runtime type → T4 GPU**, add your Hugging Face token under the key icon (**Secrets**) as `HF_TOKEN` with notebook access switched on, and choose **Runtime → Run all**. The notebook creates the Python 3.10 environment, downloads the data, runs a 20-image check, runs the full benchmark and downloads `results.zip`. Progress is saved every 200 images, so an interrupted run resumes where it stopped.

## Outputs

| File | Contents |
|---|---|
| `results/<tool>_raw.csv` | Raw tool outputs per image: predicted SMILES, confidence score (MolScribe only), runtime, errors |
| `results/<tool>_predictions.csv` | Raw outputs plus every metric per image and the structure type |
| `results/summary.json` | All metrics per tool: overall, per structure type and per official group, plus run details (date, git commit, Python, torch, RDKit versions, seed, GPU use) |
| `results/comparison.csv` | Side-by-side table of both tools, including the published BioVista scores for reference |
| `results/known_unscoreable_rows.csv` | Ground-truth rows that cannot be scored (see Limitations) |

## Metrics

| Metric | Meaning |
|---|---|
| **Official accuracy** | BioMiner's official BioVista matcher: both SMILES are neutralized, converted to RDKit canonical form, the first `[1*]` label is ignored, and the strings must be identical. Reported overall and for BioMiner's groups (Full, Chiral, Markush), so results are comparable with the published table. |
| **Exact match (InChIKey)** | Predicted and true molecules have the same InChIKey. A standard cheminformatics check; not defined for Markush structures. |
| **Exact match without stereochemistry** | Same first InChIKey block: correct atoms and bonds, ignoring chirality. The gap to exact match shows how many errors are stereochemistry only. |
| **Mean Tanimoto similarity** | Morgan fingerprint similarity (radius 2, 2,048 bits), from 0 to 1. Gives partial credit for near-misses. |
| **Valid SMILES rate** | Share of predictions RDKit can parse as a single molecule. |
| **Runtime** | Mean seconds per image. |

BioMiner's groups overlap: *Full* = no `*` (4,844 images), *Chiral* = contains `@` (2,230), *Markush* = contains `*` (4,017). About 45% of BioVista is Markush structures (templates with R-group attachment points), which is why the official matcher matters.

**Validation of the metrics:** scoring every ground-truth SMILES against itself gives an official accuracy of 0.9992, confirming the matcher handles 99.92% of the dataset. `tests/test_metrics.py` checks each metric on known cases (for example, the same molecule written two ways, enantiomers, charged and neutral forms, invalid and Markush SMILES).

## Results

Full run: all 8,861 BioVista OCSR images on a Google Colab T4 GPU, with Python 3.10, torch 2.6.0 and RDKit 2026.09.1 (details in `results/summary.json`). Both tools ran without errors.

**Official accuracy (BioMiner's matcher), compared with the published scores from the BioMiner README:**

| Tool | Full | Chiral | Markush | All |
|---|---|---|---|---|
| MolGlyph (this benchmark) | 0.757 | 0.503 | 0.770 | **0.763** |
| MolGlyph (published) | 0.758 | 0.504 | 0.770 | 0.764 |
| MolScribe (this benchmark) | 0.704 | 0.486 | 0.170 | **0.461** |
| MolScribe (published) | 0.703 | 0.481 | 0.156 | 0.455 |

**Additional metrics (all 8,861 images):**

| Tool | Exact match (InChIKey) | Exact match, no stereo | Mean Tanimoto | Valid SMILES rate | Seconds per image (T4) |
|---|---|---|---|---|---|
| MolGlyph | 0.413 | 0.466 | 0.495 | 0.531 | 0.137 |
| MolScribe | 0.382 | 0.414 | 0.467 | 0.802 | 0.202 |

RDKit-based metrics are 0 on Markush rows by design (see Limitations), so they are best read on full molecules. On the 3,242 full structures without stereochemistry, InChIKey exact match is 0.862 for MolGlyph and 0.758 for MolScribe.

### Findings

- **The pipeline reproduces the published evaluation.** MolGlyph's scores match the published values to within 0.001 in every group, and MolScribe's to within 0.006 overall. The slightly higher MolScribe Markush score (+0.014) is most likely due to the newer RDKit version used here, which can write some canonical SMILES differently.
- **MolGlyph's advantage comes from Markush structures.** It recognises 77% of Markush structures against MolScribe's 17%, and since Markush structures make up 45% of BioVista, this drives most of the overall gap (0.763 vs 0.461). On full molecules the gap is smaller (0.757 vs 0.704).
- **Stereochemistry is the weakest area for both tools.** Both score about 0.50 on chiral structures. For MolGlyph on chiral structures, exact match rises from 0.539 to 0.786 when stereochemistry is ignored, so most of its chiral errors involve only the 3D configuration, not the atoms and bonds.
- **MolGlyph is also faster**, at 0.14 seconds per image against 0.20 on a T4 GPU (about 20 and 30 minutes for the full dataset).

## Limitations and notes

- **Seven unscoreable ground-truth rows.** Seven SMILES, all containing boron, fail BioMiner's official matcher even when compared with themselves, so no prediction can be scored correct on them. They are kept, matching the published evaluation, and listed in `results/known_unscoreable_rows.csv`.
- **Validity and Tanimoto on Markush rows.** RDKit-based metrics (InChIKey exact match, Tanimoto, valid SMILES rate) treat any SMILES with `*` attachment points as not a single molecule, so they are 0 on Markush rows by design. This is also why MolGlyph's valid SMILES rate (0.531) is lower than MolScribe's: MolGlyph correctly outputs Markush structures far more often. Use the official accuracy for Markush structures; read the RDKit-based metrics mainly for full molecules.
- **MolGlyph output format.** MolGlyph outputs extended SMILES (E-SMILES) with atom annotations such as `<a>0:SEM</a>`. Each output is converted to plain SMILES with BioMiner's own `Translator.refactor`, exactly as in `BioMiner/commons/ocsr.py`; the converter (`molglyph_utils.py`, `abbrevs.csv`) is vendored unmodified in `third_party/biominer_molscribe/`.
- **RDKit version.** BioMiner used RDKit 2022.9.5, which has no Apple Silicon build; this project uses RDKit 2026.09.1. Canonical SMILES can differ slightly between versions, so scores may differ from the published ones by a small amount.
- **albumentations version.** MolScribe's image transform only works with albumentations 1.x, so this project pins 1.1.0 (the version the original MolScribe used) rather than BioMiner's 2.0.5. OpenNMT-py 2.2.0 and torchtext 0.5.0 are also required by the vendored code.
- **Checkpoint loading.** The checkpoints store training arguments, which torch 2.6 blocks by default. `src/tools/vendored.py` allows full loading for these official checkpoints only.
- **matplotlib backend.** The vendored code imports matplotlib. Colab sets a notebook-only plotting backend that the separate Python 3.10 environment lacks, so `src/tools/vendored.py` sets the non-display `Agg` backend.
- **Confidence scores.** MolScribe's confidence score did not reliably separate correct from incorrect predictions in early tests, so it is reported but not used to filter results. MolGlyph does not report a confidence score.
- **Hardware.** The full benchmark takes about 50 minutes for both tools on a Colab T4 GPU. CPU runs are practical only for small samples (about 1.5–2.5 seconds per image on an Apple M-series laptop).

## Credits and references

- BioMiner and BioVista: Yan, J., Zhu, J., Yang, Y., Liu, Q., Zhang, K., Zhang, Z., Liu, X., Zhang, B., Gao, K., Xiao, J., & Chen, E. (2025). *BioMiner: A multi-modal system for automated mining of protein-ligand bioactivity data from literature.* bioRxiv. https://doi.org/10.1101/2025.04.22.648951 · Code: https://github.com/jiaxianyan/BioMiner (MIT license)
- MolScribe: Qian, Y., Guo, J., Tu, Z., Li, Z., Coley, C. W., & Barzilay, R. (2023). MolScribe: Robust molecular structure recognition with image-to-graph generation. *Journal of Chemical Information and Modeling, 63*(7), 1925–1934. https://doi.org/10.1021/acs.jcim.2c01480
- The code in `third_party/biominer_molscribe/` is BioMiner's, included unmodified under its MIT license, together with the original MolScribe license. The source commit is recorded in that folder's README.
