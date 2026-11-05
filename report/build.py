"""Build report/report.pdf: resolve FIG(key) references to figure numbers, then run pandoc.

A figure's key is the file-name stem of its image. Figures are numbered in order of
appearance, which is the order LaTeX uses, so the numbers in the text match the captions.
"""
import os
import re
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(here, "report.md")
text = open(src).read()
stems = []
for m in re.finditer(r"^!\[[^\]]*\]\(([^)]+)\)", text, flags=re.M):
    stems.append(os.path.splitext(os.path.basename(m.group(1)))[0])
numbers = {}
for i, s in enumerate(stems, start=1):
    if s in numbers:
        sys.exit(f"duplicate figure stem {s}")
    numbers[s] = i
unresolved = [k for k in re.findall(r"FIG\(([^)]+)\)", text) if k not in numbers]
if unresolved:
    sys.exit(f"unresolved figure keys: {sorted(set(unresolved))}")
left = re.findall(r"\u00ab[^\u00bb]*\u00bb", text)
if left:
    sys.exit(f"unfilled markers: {left[:10]}")
out = re.sub(r"FIG\(([^)]+)\)", lambda m: str(numbers[m.group(1)]), text)
build_md = os.path.join(here, "report_build.md")
open(build_md, "w").write(out)
cmd = ["pandoc", build_md, "-o", os.path.join(here, "report.pdf"), "--pdf-engine=xelatex",
       f"--resource-path={here}:{os.path.dirname(here)}", "-V", "linkcolor=blue"]
subprocess.run(cmd, check=True)
os.remove(build_md)
print(f"wrote report/report.pdf with {len(stems)} figures")
