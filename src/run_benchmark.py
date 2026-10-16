"""Run the full OCSR benchmark: data loading -> inference -> metrics -> output.

Usage (from the project root):
    python src/run_benchmark.py                          # all tools, all images
    python src/run_benchmark.py --tools molscribe --limit 5
    python src/run_benchmark.py --limit 500 --seed 42    # reproducible sample

Outputs (in results/ by default):
    <tool>_raw.csv          raw tool outputs, written incrementally (lets a run resume)
    <tool>_predictions.csv  raw outputs plus per-image metric columns
    summary.json            all metrics per tool, overall and per structure type, plus run info
    comparison.csv          side-by-side table of the main metrics, with published reference values
"""
import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so 'src' imports work when run as a script

from src.data import load_biovista_ocsr  # noqa: E402
from src.metrics import evaluate_table, summarize  # noqa: E402
from src.tools import build_tool  # noqa: E402

RAW_COLUMNS = ["image_path", "image_name", "true_smiles", "pred_smiles",
               "confidence", "runtime_s", "error"]


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark OCSR tools on BioVista.")
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    p.add_argument("--tools", nargs="+", default=["molscribe", "molglyph"])
    p.add_argument("--limit", type=int, default=None,
                   help="Use a random sample of N images (overrides config sample_n)")
    p.add_argument("--seed", type=int, default=None, help="Sampling seed (overrides config)")
    p.add_argument("--fresh", action="store_true",
                   help="Ignore saved progress and rerun every image")
    return p.parse_args()


def run_inference(tool_name, pairs, cfg, raw_path, fresh):
    """Run one tool on all pairs, appending results to raw_path every few images.

    If raw_path already exists (e.g. after a Colab disconnect), images already
    processed are skipped, so the run continues where it stopped.
    """
    bench = cfg["benchmark"]
    done = set()
    if raw_path.exists() and not fresh:
        done = set(pd.read_csv(raw_path)["image_path"])
    elif raw_path.exists():
        raw_path.unlink()
    todo = [(img, smi) for img, smi in pairs if img not in done]
    print(f"[{tool_name}] {len(done)} images already done, {len(todo)} to run")
    if not todo:
        return

    tool = build_tool(tool_name, cfg["tools"][tool_name], bench)
    print(f"[{tool_name}] running on {tool.device}")
    step = bench["checkpoint_every"]
    for start in tqdm(range(0, len(todo), step), desc=tool_name, unit="chunk"):
        chunk = todo[start:start + step]
        preds = tool.predict([img for img, _ in chunk])
        rows = [{"image_path": p.image_path, "image_name": Path(p.image_path).name,
                 "true_smiles": smi, "pred_smiles": p.pred_smiles,
                 "confidence": p.confidence, "runtime_s": p.runtime_s, "error": p.error}
                for p, (_, smi) in zip(preds, chunk)]
        pd.DataFrame(rows, columns=RAW_COLUMNS).to_csv(
            raw_path, mode="a", header=not raw_path.exists(), index=False)


def evaluate_tool(tool_name, pairs, raw_path, out_dir):
    """Score saved predictions for the images in this run and save the scored table."""
    raw = pd.read_csv(raw_path)
    wanted = {img for img, _ in pairs}
    raw = raw[raw["image_path"].isin(wanted)].drop_duplicates("image_path", keep="last")
    scored = evaluate_table(raw, pred_col="pred_smiles", true_col="true_smiles")
    scored.to_csv(out_dir / f"{tool_name}_predictions.csv", index=False)

    summary = summarize(scored)
    summary["runtime"] = {"mean_s_per_image": float(raw["runtime_s"].mean()),
                          "total_s": float(raw["runtime_s"].sum())}
    summary["n_errors"] = int(raw["error"].notna().sum())
    return summary


def comparison_table(summaries, published):
    """One row per tool: the headline metrics side by side, plus published values."""
    rows = []
    for tool, s in summaries.items():
        groups = s["official_groups"]
        ref = published.get(tool, {})
        rows.append({
            "tool": tool,
            "n_images": s["overall"]["n"],
            "official_accuracy_all": s["overall"]["official_match"],
            "official_accuracy_full": groups["Full"]["official_match"],
            "official_accuracy_chiral": groups["Chiral"]["official_match"],
            "official_accuracy_markush": groups["Markush"]["official_match"],
            "exact_match_inchikey": s["overall"]["exact_match"],
            "exact_match_no_stereo": s["overall"]["exact_match_no_stereo"],
            "mean_tanimoto": s["overall"]["tanimoto"],
            "valid_smiles_rate": s["overall"]["is_valid"],
            "mean_runtime_s_per_image": s["runtime"]["mean_s_per_image"],
            "n_errors": s["n_errors"],
            "published_all": ref.get("All"),
            "published_full": ref.get("Full"),
            "published_chiral": ref.get("Chiral"),
            "published_markush": ref.get("Markush"),
        })
    return pd.DataFrame(rows)


def run_info(cfg, n_images, seed):
    """Details needed to reproduce this run."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = None
    import torch
    import rdkit
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": commit, "python": platform.python_version(),
            "torch": torch.__version__, "rdkit": rdkit.__version__,
            "cuda_available": torch.cuda.is_available(),
            "n_images": n_images, "seed": seed, "device_setting": cfg["benchmark"]["device"]}


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    bench = cfg["benchmark"]
    sample_n = args.limit if args.limit is not None else bench["sample_n"]
    seed = args.seed if args.seed is not None else bench["seed"]

    out_dir = ROOT / bench["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data loading
    pairs = load_biovista_ocsr(Path(args.config), sample_n=sample_n, seed=seed)

    summaries = {}
    for tool_name in args.tools:
        raw_path = out_dir / f"{tool_name}_raw.csv"
        # 2. Inference (resumable)
        run_inference(tool_name, pairs, cfg, raw_path, args.fresh)
        # 3. Metrics
        summaries[tool_name] = evaluate_tool(tool_name, pairs, raw_path, out_dir)

    # 4. Output
    table = comparison_table(summaries, cfg.get("published", {}))
    table.to_csv(out_dir / "comparison.csv", index=False)
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"run_info": run_info(cfg, len(pairs), seed), "tools": summaries}, f, indent=2)

    print("\n" + table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved results to {out_dir}")


if __name__ == "__main__":
    main()
