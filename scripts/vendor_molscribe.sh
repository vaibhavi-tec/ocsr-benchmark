#!/usr/bin/env bash
# Copies BioMiner's bundled MolScribe/MolGlyph inference code into third_party/.
# Run once from the project root:  bash scripts/vendor_molscribe.sh
set -euo pipefail

REPO=https://github.com/jiaxianyan/BioMiner
DEST=third_party/biominer_molscribe
TMP=$(mktemp -d)

# Sparse, shallow clone: downloads only the MolScribe folder, not BioMiner's large model files
git clone --quiet --depth 1 --filter=blob:none --sparse "$REPO" "$TMP"
git -C "$TMP" sparse-checkout set BioMiner/MolScribe
COMMIT=$(git -C "$TMP" rev-parse HEAD)

rm -rf "$DEST" && mkdir -p "$DEST"
cp -R "$TMP/BioMiner/MolScribe/molscribe" "$DEST/"
cp "$TMP/LICENSE" "$DEST/LICENSE"
# Linux-only native Indigo binaries are only used for training-data rendering, not inference
rm -rf "$DEST/molscribe/indigo/lib"

cat > "$DEST/README.md" <<NOTE
# Vendored code: BioMiner's MolScribe / MolGlyph inference package

- Source: $REPO (folder \`BioMiner/MolScribe/molscribe\`)
- Commit: $COMMIT
- License: MIT (see LICENSE in this folder), copyright the BioMiner authors
- Changes: none to the Python code. The Linux-only Indigo binaries in
  \`molscribe/indigo/lib\` were removed; they are only needed for generating
  training images, not for inference.

This code is not part of this project's own work. It is included unmodified so
that both MolScribe and MolGlyph run from the same codebase, and so the
benchmark keeps working if the upstream repository changes.
NOTE

rm -rf "$TMP"
echo "Vendored BioMiner MolScribe code at commit $COMMIT into $DEST"
