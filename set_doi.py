#!/usr/bin/env python3
"""
Back-fill the Zenodo DOI (and, optionally, the GitHub URL) after the first
release, so citation metadata stops carrying placeholders.

Zenodo mints the DOI when the release is archived, which is necessarily after
the tag exists -- so v1.0.0 ships without it and v1.0.1 carries it. This script
does that edit across README.md, CITATION.cff and .zenodo.json in one shot.

    python3 set_doi.py 10.5281/zenodo.1234567
    python3 set_doi.py 10.5281/zenodo.1234567 --repo https://github.com/USER/exact-null-metrology

Use the **concept DOI** (the one Zenodo labels "all versions") -- it always
resolves to the latest version, which is what a citation should point at.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def read(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as fh:
        return fh.read()


def write(name, text):
    with open(os.path.join(HERE, name), "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"  updated {name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("doi", help="concept DOI, e.g. 10.5281/zenodo.1234567")
    ap.add_argument("--repo", help="public repository URL")
    a = ap.parse_args()

    doi = a.doi.strip().removeprefix("https://doi.org/")
    if not re.fullmatch(r"10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+", doi):
        print(f"does not look like a DOI: {doi}")
        return 2
    print(f"setting DOI {doi}")

    # ---- CITATION.cff: replace the commented placeholders with real keys
    cff = read("CITATION.cff")
    cff = re.sub(r"^# Filled in after.*\n", "", cff, flags=re.M)
    cff = re.sub(r'^# doi:.*$', f'doi: "{doi}"', cff, flags=re.M)
    if "\ndoi:" not in cff:
        cff = cff.rstrip("\n") + f'\ndoi: "{doi}"\n'
    if a.repo:
        cff = re.sub(r'^# repository-code:.*$',
                     f'repository-code: "{a.repo}"', cff, flags=re.M)
        if "\nrepository-code:" not in cff:
            cff = cff.rstrip("\n") + f'\nrepository-code: "{a.repo}"\n'
    write("CITATION.cff", cff)

    # ---- .zenodo.json: record the DOI so later versions link back
    zen = json.loads(read(".zenodo.json"))
    zen["doi"] = doi
    write(".zenodo.json", json.dumps(zen, indent=2, ensure_ascii=False) + "\n")

    # ---- README: a badge + an explicit citation line under the title
    rd = read("README.md")
    badge = (f"[![DOI](https://zenodo.org/badge/DOI/{doi}.svg)]"
             f"(https://doi.org/{doi})\n")
    rd = re.sub(r"^\[!\[DOI\].*\n", "", rd, flags=re.M)
    lines = rd.split("\n")
    for i, ln in enumerate(lines):
        if ln.startswith("# "):
            lines.insert(i + 1, "\n" + badge.rstrip())
            break
    rd = "\n".join(lines)
    # Match the whole sentence through its known terminator, across line
    # breaks, so neither a wrap nor the dot in "CITATION.cff" splits it.
    rd, n = re.subn(r"Cite the archived record;.*?the DOI\.",
                    f"Cite the archived record — DOI "
                    f"[{doi}](https://doi.org/{doi}). "
                    f"`CITATION.cff` carries the machine-readable form.",
                    rd, count=1, flags=re.S)
    if not n:
        print("  ! README citation sentence not found; edit it by hand")
    write("README.md", rd)

    print("\ndone. Now:  git commit -am 'Record Zenodo DOI' && "
          "git tag -a v1.0.1 -m 'Record DOI' && git push --follow-tags")
    return 0


if __name__ == "__main__":
    sys.exit(main())
