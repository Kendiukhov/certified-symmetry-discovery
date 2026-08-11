# Submission bundle — *Machine Learning* (Springer)

Everything the journal needs, formatted with the official Springer Nature LaTeX
template. Upload the contents of this folder (or the zip built from it) together
with `main.pdf` and `cover_letter.pdf`.

## What to upload

| file | purpose |
|---|---|
| `main.pdf` | the compiled manuscript |
| `main.tex` | manuscript source |
| `sections/*.tex` | the body, `\input` by `main.tex` |
| `tables/*.tex` | table bodies, generated from the raw results |
| `figures/*.pdf` | figures, vector PDF with fonts embedded |
| `refs.bib` | bibliography |
| `sn-jnl.cls` | Springer Nature class (December 2024 release) |
| `sn-apacite.bst` | the APA-style bibliography style selected by `sn-apa` |
| `cover_letter.pdf`, `cover_letter.tex` | cover letter |

## How it meets the journal's requirements

- **Template.** Springer Nature `sn-jnl.cls`, single column, option `sn-apa`, so
  references are cited by name and year and formatted in APA style with DOIs
  given as full links. The `lineno` option numbers lines for reviewers; drop it
  if the editors prefer otherwise.
- **Title page.** Title, author, affiliation with department, city and country,
  and the corresponding author's email. No ORCID is on file; add one with
  `\orcid{...}` after `\sur{Kendiukhov}` if you have it.
- **Abstract.** 248 words, inside the journal's 150–250 range, with no
  equations, citations, subheadings or undefined abbreviations.
- **Keywords.** Six, as required (4–6).
- **Statements and Declarations.** Funding, competing interests, ethics, consent,
  data availability, materials availability, code availability and author
  contribution, in the order Springer lists them.
- **Figures.** Vector PDF at final size, fonts embedded (checked with
  `pdffonts`), lettering in a sans face at roughly 2–3 mm, and all captions in
  the manuscript rather than in the image files.

## Two things to confirm before you submit

The declarations in `main.tex` are marked with `% CHECK` where they assert
something only the author can confirm:

1. **Funding** — currently states that no funds, grants or other support were
   received. Change it if any grant applies.
2. **Competing interests** — currently states that there are none.

## Rebuilding

```bash
./build.sh          # refresh from ../paper and ../figures, then compile
```

`build.sh` copies the section text, tables, figures and bibliography from the
main tree rather than keeping a second copy under edit, so the submitted
manuscript and the repository version cannot drift apart. Edit the originals in
`../paper/`, not the copies here.

The only content that differs between the two versions is the front matter:
the submission carries the Springer title block, keywords and declarations,
while `../paper/main.pdf` is the plain preprint. The body, figures, tables and
bibliography are identical files.
