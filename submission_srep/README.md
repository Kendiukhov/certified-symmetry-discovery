# Revision bundle — *Scientific Reports*

Revised manuscript for submission 43fb9b81-0ebd-4a9a-a390-5ba988a88425,
formatted with the official Springer Nature LaTeX template (`sn-jnl.cls`,
December 2024 release) and the Nature reference style (`sn-nature`).
`build.sh` regenerates everything here from the sources in `paper/`,
`figures/` and this folder, so the preprint and the submission cannot diverge.

## What to upload

| file | purpose |
|---|---|
| `main.tex` (+ `sections/`, `tables/`, `refs.bib`, `main.bbl`, `sn-jnl.cls`, `sn-nature.bst`) | revised manuscript, LaTeX (the journal does not accept PDF article text for revisions) |
| `main.pdf` | compiled manuscript, for checking |
| `figures/*.pdf` | each figure as a separate vector file (the number in each file name is the figure number: `fig1_example` … `fig5_selection` = Figs. 1–5, `figS1_dependence` … `figS6_invariance` = Supplementary Figs. S1–S6) |
| `supplementary.pdf` (+ `supplementary.tex`) | Supplementary Information, a single separate file |
| `response_to_reviewers.pdf` | point-by-point response |
| `cover_letter.pdf` | cover letter |
| `submission.zip` | all of the above in one archive (`./build.sh zip`) |

## How it meets the journal's requirements

- **Title** 17 words; **abstract** unstructured, no citations, under 200 words
  (the build prints the count).
- **Structure**: Introduction, Results (with subheadings), Discussion, Methods;
  Data availability and Code availability before the references; Acknowledgements,
  Author contributions and Additional information (competing interests) after.
- **References** numbered in square brackets, Nature style.
- **Display items**: five figures and three tables in the main text (limit 8);
  everything else is in the Supplementary Information, numbered S1, S2, ...
  The build checks that the supplementary numbers cited in the main text match
  the compiled Supplementary Information.
- **Revised-manuscript format**: single column, unjustified text, line and page
  numbers, Computer Modern fonts, no footnotes, figures also supplied as
  separate files.
- **Figure legends** begin with a title sentence and stay under 350 words;
  lettering is sans-serif with bold lower-case panel letters.

## To confirm before submitting

1. The funding/acknowledgements and competing-interest declarations in
   `main.tex` and the language-model statement at the end of "Software,
   reproducibility and compute" in `paper/sections/methods.tex` are written
   out; they need the author's confirmation.
2. Optionally archive the tagged repository version (`v2.0.0`) on Zenodo and
   add the DOI to the Code availability statement.
