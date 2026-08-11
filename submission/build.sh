#!/bin/bash
# Refresh the submission bundle from the single source of truth and compile it.
#
# The section text, bibliography, generated tables and figures live in paper/
# and figures/. They are copied here rather than edited here, so the two
# versions of the manuscript cannot drift apart. Edit the originals.
set -euo pipefail
cd "$(dirname "$0")"

rm -rf sections tables figures
mkdir -p sections tables figures
cp ../paper/sections/*.tex sections/
cp ../paper/tables/*.tex tables/
cp ../paper/refs.bib .
for f in ../figures/*.pdf; do cp "$f" figures/; done

# The submission is self-contained, so figure paths lose the ../ prefix.
sed -i '' 's|{\.\./figures/|{figures/|g' sections/*.tex 2>/dev/null \
  || sed -i 's|{\.\./figures/|{figures/|g' sections/*.tex

# The Springer class provides the appendix environment, so drop the article
# class's \appendix switch if one is present in the copied sources.
sed -i '' 's|^\\appendix$||' sections/*.tex 2>/dev/null \
  || sed -i 's|^\\appendix$||' sections/*.tex

pdflatex -interaction=nonstopmode main.tex >/dev/null
bibtex main >/dev/null || true
pdflatex -interaction=nonstopmode main.tex >/dev/null
pdflatex -interaction=nonstopmode main.tex >build.log 2>&1

pdflatex -interaction=nonstopmode cover_letter.tex >/dev/null
pdflatex -interaction=nonstopmode cover_letter.tex >/dev/null

echo "errors:   $(grep -c '^! ' build.log || true)"
echo "warnings: $(grep -c 'Warning: \(Citation\|Reference\)' build.log || true)"
echo "pages:    $(pdfinfo main.pdf | awk '/Pages/{print $2}')"

if [ "${1:-}" = "zip" ]; then
  rm -f submission.zip
  # main.bbl is included so the bundle typesets even where bibtex is not run.
  # -x drops the AppleDouble and .DS_Store files that macOS leaves on this
  # volume; a submission system would otherwise see them as stray sources.
  zip -qr submission.zip main.tex main.bbl main.pdf sections tables figures \
      refs.bib sn-jnl.cls sn-apacite.bst cover_letter.tex cover_letter.pdf \
      README.md -x '._*' '*/._*' '.DS_Store' '*/.DS_Store' 
  echo "wrote $(pwd)/submission.zip"
fi
