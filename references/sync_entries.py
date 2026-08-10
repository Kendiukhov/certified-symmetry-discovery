"""Regenerate the ENTRIES list in verify_refs.py from paper/refs.bib.

Keeps the verification report in exact correspondence with what the paper cites.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    bib = (ROOT / "paper" / "refs.bib").read_text()
    out = []
    for key, body in re.findall(r"@\w+\{([^,]+),(.*?)\n\}", bib, re.S):
        title = re.search(r"title\s*=\s*\{(.*?)\},?\n", body, re.S).group(1)
        title = " ".join(title.split()).replace("{", "").replace("}", "").replace("\\\\", "")
        out.append((key.strip(), title))
    lines = ",\n".join(f"    ({k!r}, {t!r})" for k, t in out)
    path = ROOT / "references" / "verify_refs.py"
    s = path.read_text()
    i = s.index("ENTRIES = [")
    j = s.index("\n]\n", i)
    path.write_text(s[:i] + "# Generated from paper/refs.bib by references/sync_entries.py.\nENTRIES = [\n"
                    + lines + s[j:])
    print(f"synced {len(out)} entries")


if __name__ == "__main__":
    main()
