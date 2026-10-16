"""Quick checks for src/metrics.py. Run: python tests/test_metrics.py"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import metrics as m

ETHANOL_A, ETHANOL_B = "CCO", "OCC"
L_ALA, D_ALA = "C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O"


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r}, want {want!r}")
    return ok


results = [
    check("same molecule, two spellings -> exact_match", m.exact_match(ETHANOL_B, ETHANOL_A), True),
    check("different molecules -> no exact_match", m.exact_match("CCN", ETHANOL_A), False),
    check("enantiomers -> not exact_match", m.exact_match(D_ALA, L_ALA), False),
    check("enantiomers -> exact_match_no_stereo", m.exact_match_no_stereo(D_ALA, L_ALA), True),
    check("garbage SMILES -> not valid", m.is_valid("not_a_smiles"), False),
    check("valid SMILES -> valid", m.is_valid("c1ccccc1"), True),
    check("empty/None prediction -> not valid", (m.is_valid(""), m.is_valid(None)), (False, False)),
    check("tanimoto identical = 1.0", m.tanimoto(ETHANOL_B, ETHANOL_A), 1.0),
    check("tanimoto invalid = 0.0", m.tanimoto("garbage", ETHANOL_A), 0.0),
    check("tanimoto partial in (0,1)", 0 < m.tanimoto("CCCO", ETHANOL_A) < 1, True),
    check("Markush '[1*]' flagged", m.is_markush("[1*]C(=O)O"), True),
    check("'[R1]' flagged and no crash", (m.is_markush("[R1]CC"), m.exact_match("[R1]CC", ETHANOL_A)), (True, False)),
    # --- markush_match (BioMiner's official matcher) ---
    check("markush_match: '[1*]' label ignored vs '*'", m.markush_match("*CC", "[1*]CC"), True),
    check("markush_match: same R-group written two ways", m.markush_match("[2*]C(=O)c1ccccc1", "O=C([2*])c1ccccc1"), True),
    check("markush_match: different R-group label '[2*]' vs '[3*]'", m.markush_match("[2*]CC", "[3*]CC"), False),
    check("markush_match: different scaffold", m.markush_match("*CCC", "*CC"), False),
    check("markush_match: plain molecule, two spellings", m.markush_match("OCC", "CCO"), True),
    check("markush_match: garbage / None -> False, no crash", (m.markush_match("garbage", "*CC"), m.markush_match(None, "*CC")), (False, False)),
    check("markush_match: charged form neutralized", m.markush_match("C[N+](C)(C)[O-]", "C[N+](C)(C)[O-]"), True),
    # --- structure_type ---
    check("structure_type markush/chiral/full", (m.structure_type("[1*]CC"), m.structure_type("C[C@H](N)O"), m.structure_type("CCO")), ("markush", "chiral", "full")),
    check("structure_type chiral Markush -> markush", m.structure_type("*[C@H](N)O"), "markush"),
    check("structure_type skipped rows -> None", (m.structure_type("c"), m.structure_type(None)), (None, None)),
    check("Markush counts as not valid", (m.is_valid("*CC"), m.is_valid("[R1]CC")), (False, False)),
    check("Markush vs identical Markush -> no match, no crash", m.evaluate_pair("*CC", "*CC")["exact_match"], False),
]

df = pd.DataFrame({"pred_smiles": [ETHANOL_B, D_ALA, "garbage", "*CC", "CCO"],
                   "true_smiles": [ETHANOL_A, L_ALA, "CCN", "[1*]CC", "c"]})
res = m.evaluate_table(df)
print("\n", res.to_string(), "\n")
summ = m.summarize(res)
print("summary:", summ)
check("official_match per row", res["official_match"].tolist(), [True, False, False, True, None])
check("summary n (skipped 'c' row excluded)", (summ["overall"]["n"], summ["n_excluded"]), (4, 1))
check("overall official accuracy", summ["overall"]["official_match"], 0.5)
check("by_type counts", {k: v["n"] for k, v in summ["by_type"].items()}, {"chiral": 1, "full": 2, "markush": 1})
check("markush official accuracy", summ["by_type"]["markush"]["official_match"], 1.0)
check("official_groups: Chiral n / Markush n", (summ["official_groups"]["Chiral"]["n"], summ["official_groups"]["Markush"]["n"]), (1, 1))

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
