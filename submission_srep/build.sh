#!/bin/bash
# Refresh the Scientific Reports bundle from the single source of truth and
# compile the manuscript, the Supplementary Information, the response to the
# reviewers and the cover letter.
#
# The section text, bibliography, generated tables and figures live in paper/,
# figures/ and revision/. They are copied here rather than edited here, so the
# preprint and the submission cannot drift apart. Edit the originals.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}

rm -rf sections tables figures
mkdir -p sections tables figures
cp ../paper/sections/*.tex sections/
cp ../paper/tables/*.tex tables/
cp ../paper/refs.bib .
MAIN_FIGS="fig1_example fig2_headline fig3_power fig4_mechanisms fig5_selection"
SI_FIGS="figS1_dependence figS2_modelerror figS3_eiv figS4_tolerance figS5_tightness figS6_invariance"
for f in $MAIN_FIGS $SI_FIGS; do cp "../figures/$f.pdf" figures/; done

compile () {   # name
  pdflatex -interaction=nonstopmode "$1.tex" >/dev/null || true
  if grep -q '\\bibdata' "$1.aux" 2>/dev/null; then bibtex "$1" >/dev/null || true; fi
  pdflatex -interaction=nonstopmode "$1.tex" >/dev/null || true
  pdflatex -interaction=nonstopmode "$1.tex" > "$1.build.log" 2>&1 || true
}
compile main
compile supplementary

# Supplementary items are cited in the main text through macros with fixed
# numbers; check that the compiled Supplementary Information agrees.
$PY - <<'EOF'
import re, sys, pathlib
macros = pathlib.Path("sections/macros.tex").read_text()
want = {}
for name, kind, num in re.findall(r"\\newcommand\{\\(S[FTN]\w+)\}\{Supplementary (Fig\.|Table|Note)~S?(\d+)\}", macros):
    want[name] = (kind, num)
aux = pathlib.Path("supplementary.aux").read_text()
got = dict(re.findall(r"\\newlabel\{(s(?:fig|tab):\w+)\}\{\{S?(\d+)\}", aux))
bad = 0
for name, (kind, num) in want.items():
    if kind == "Note":
        continue
    key = ("sfig:" if name.startswith("SF") else "stab:") + name[2:]
    if got.get(key) != num:
        print(f"SI numbering mismatch: {name} says {num}, compiled {got.get(key)}")
        bad += 1
print("SI numbering:", "consistent" if not bad else f"{bad} mismatches")
sys.exit(1 if bad else 0)
EOF

for d in main supplementary; do
  echo "$d: errors $(grep -c '^! ' $d.build.log || true), undefined refs/cites" \
       "$(grep -c 'Warning: \(Citation\|Reference\).*undefined' $d.build.log || true)," \
       "pages $(pdfinfo $d.pdf | awk '/Pages/{print $2}')"
done
echo "abstract words: $(detex sections/abstract.tex | wc -w | tr -d ' ')"
echo "references: $(grep -c '^\\bibitem' main.bbl || true)"
echo "pending markers in main: $(grep -c 'pending{' sections/*.tex | awk -F: '{s+=$2} END {print s}')"

# Response letter and cover letter.
for doc in response_to_reviewers cover_letter; do
  if [ -f "$doc.tex" ]; then
    pdflatex -interaction=nonstopmode "$doc.tex" >/dev/null || true
    pdflatex -interaction=nonstopmode "$doc.tex" > "$doc.build.log" 2>&1 || true
    echo "$doc: errors $(grep -c '^! ' $doc.build.log || true), pages $(pdfinfo $doc.pdf | awk '/Pages/{print $2}')"
  fi
done

if [ "${1:-}" = "zip" ]; then
  rm -f submission.zip
  # main.bbl is included so the bundle typesets where bibtex is not run; the
  # figures are also supplied as separate files, as required for revisions.
  zip -qr submission.zip main.tex main.bbl main.pdf supplementary.tex supplementary.pdf \
      sections tables figures refs.bib sn-jnl.cls sn-nature.bst \
      response_to_reviewers.tex response_body.tex response_to_reviewers.pdf cover_letter.tex cover_letter.pdf \
      README.md -x '._*' '*/._*' '.DS_Store' '*/.DS_Store'
  echo "wrote $(pwd)/submission.zip"
fi
