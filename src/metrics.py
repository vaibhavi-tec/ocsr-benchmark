"""OCSR metrics comparing a predicted SMILES with the true SMILES (RDKit)."""
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import inchi, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")  # don't spam stderr for invalid SMILES

# Morgan fingerprint generator (radius 2, 2048 bits), created once and reused
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def parse(smiles):
    """Return an RDKit Mol, or None if the SMILES is missing/invalid/Markush.

    RDKit accepts '*' as a wildcard atom, but a Markush structure is not a
    single concrete molecule, so it is treated as not parseable (and flagged
    separately via is_markush).
    """
    if not isinstance(smiles, str) or not smiles.strip() or is_markush(smiles):
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def is_markush(smiles) -> bool:
    """Flag Markush-style SMILES: wildcard atoms ('*', '[1*]') or [R1]-style labels.

    Checked on the text so it also works when RDKit can't parse the string.
    """
    if not isinstance(smiles, str):
        return False
    return "*" in smiles or "[R" in smiles


def is_valid(pred) -> bool:
    """RDKit can parse the prediction."""
    return parse(pred) is not None


def _inchikey(smiles):
    """InChIKey of a SMILES, or None if it can't be parsed or converted."""
    mol = parse(smiles)
    if mol is None:
        return None
    try:
        key = inchi.MolToInchiKey(mol)
    except Exception:
        return None
    return key or None


def exact_match(pred, true) -> bool:
    """Identical InChIKey (includes stereochemistry)."""
    kp, kt = _inchikey(pred), _inchikey(true)
    return kp is not None and kp == kt


def exact_match_no_stereo(pred, true) -> bool:
    """Identical first InChIKey block (connectivity only, ignores stereo)."""
    kp, kt = _inchikey(pred), _inchikey(true)
    return kp is not None and kt is not None and kp.split("-")[0] == kt.split("-")[0]


def tanimoto(pred, true) -> float:
    """Morgan fingerprint (radius 2, 2048 bits) Tanimoto similarity; 0.0 if unparseable."""
    mp, mt = parse(pred), parse(true)
    if mp is None or mt is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(
        _MORGAN.GetFingerprint(mp), _MORGAN.GetFingerprint(mt)))


# --- Official BioMiner matching (reproduces BioMiner/commons/process_mol.py) ----------
# BioMiner has no special Markush logic: every row (Markush or not) is scored by the
# same string comparison, applied to RDKit canonical SMILES where '*' is a wildcard atom.

_CHARGE_PATTERN = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")


def _neutralize(smiles: str) -> str:
    """Neutralize charged atoms that are not part of a +/- pair (nitro stays), return SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    for (idx,) in mol.GetSubstructMatches(_CHARGE_PATTERN):
        atom = mol.GetAtomWithIdx(idx)
        chg = atom.GetFormalCharge()
        hcount = atom.GetTotalNumHs()
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(hcount - chg)
        atom.UpdatePropertyCache()
    return Chem.MolToSmiles(mol)


def _replace_group(smiles: str) -> str:
    """Un-label the first '[1*]' wildcard atom so '[1*]' and '*' compare equal.

    Only the first isotope-1 wildcard is changed (the official code breaks after one).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "*" and atom.GetIsotope() == 1:
            atom.SetIsotope(0)
            break
    return Chem.MolToSmiles(mol)


def official_smiles_match(pred, true) -> bool:
    """BioMiner's check_smiles_match, used for every row: neutralize charges, canonicalize, drop the
    '1' label from the first R-group, then require identical strings.
    Any failure (None, invalid SMILES) counts as a mismatch.
    """
    try:
        canon = lambda smi: Chem.MolToSmiles(Chem.MolFromSmiles(_neutralize(smi)), canonical=True)
        return _replace_group(canon(pred)) == _replace_group(canon(true))
    except Exception:
        return False


def structure_type(true):
    """Category of a ground-truth SMILES, following BioMiner's OCSREvaluator.

    'markush' if it contains '*', else 'chiral' if it contains '@', else 'full'.
    Returns None for rows the official evaluator skips (missing SMILES or 'c').
    BioMiner's own counts overlap (its "chiral" includes chiral Markush and its
    "full" includes chiral full structures); see summarize()['official_groups'].
    """
    if not isinstance(true, str) or true.lower() == "c":
        return None
    if "*" in true:
        return "markush"
    return "chiral" if "@" in true else "full"


def evaluate_pair(pred, true) -> dict:
    """All metrics for one prediction, plus Markush flags for both sides."""
    return {
        "is_valid": is_valid(pred),
        "exact_match": exact_match(pred, true),
        "exact_match_no_stereo": exact_match_no_stereo(pred, true),
        "tanimoto": tanimoto(pred, true),
        "pred_is_markush": is_markush(pred),
        "true_is_markush": is_markush(true),
        "true_parseable": is_valid(true),
        "structure_type": structure_type(true),
        # Official score: BioMiner's matcher on every row (None for rows it skips)
        "official_match": official_smiles_match(pred, true) if structure_type(true) else None,
    }


def evaluate_table(df: pd.DataFrame, pred_col="pred_smiles", true_col="true_smiles") -> pd.DataFrame:
    """Add per-row metric columns to a results table."""
    rows = [evaluate_pair(p, t) for p, t in zip(df[pred_col], df[true_col])]
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


_METRICS = ["is_valid", "exact_match", "exact_match_no_stereo", "tanimoto", "official_match"]


def _means(df: pd.DataFrame) -> dict:
    out = {c: float(df[c].astype(float).mean()) for c in _METRICS}
    out["n"] = len(df)
    return out


def summarize(results: pd.DataFrame) -> dict:
    """Metric averages overall and per structure type.

    Rows the official evaluator skips (structure_type missing) are excluded and
    counted in n_excluded. Returns:
      overall         - every metric over all scored rows; official_match is the
                        overall official accuracy
      by_type         - the same metrics for 'full', 'chiral' and 'markush'
                        (mutually exclusive: markush > chiral > full)
      official_groups - BioMiner's overlapping groups as in the published table:
                        Full = no '*', Chiral = has '@', Markush = has '*'
      flags           - share of Markush / unparseable ground truths and predictions
    """
    scored = results[results["structure_type"].notna()]
    true = scored["true_smiles"]
    groups = {"Full": ~true.str.contains("*", regex=False),
              "Chiral": true.str.contains("@", regex=False),
              "Markush": true.str.contains("*", regex=False)}
    return {
        "overall": _means(scored),
        "by_type": {t: _means(g) for t, g in scored.groupby("structure_type")},
        "official_groups": {name: _means(scored[mask]) for name, mask in groups.items()},
        "flags": {c: float(scored[c].mean()) for c in
                  ["pred_is_markush", "true_is_markush", "true_parseable"]},
        "n_excluded": len(results) - len(scored),
    }
