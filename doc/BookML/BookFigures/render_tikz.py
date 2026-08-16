"""Render the TikZ schematics of the chapters to PNG for the Jupyter-book.

The book itself typesets these diagrams directly from the chapter source; the
notebooks cannot, so each figure environment containing a tikzpicture is
extracted, compiled standalone and written to
BookFigures/chapterNN_.../<label>.png, where <label> is the part of
\\label{fig:...} after the colon.  tex_to_notebook.py looks for exactly that name.

Usage:  python3 render_tikz.py [chapter numbers, default 11]
"""
import re, os, subprocess, sys, shutil, tempfile

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "..")
DIRS = {11: "chapter11_recurrent_networks"}

HEAD = r"""\documentclass[border=4pt]{standalone}
\usepackage{amsmath,amssymb,bm}
\usepackage{tikz}
\usetikzlibrary{shapes,arrows,arrows.meta,chains,positioning,fit,decorations.pathreplacing}
\begin{document}
"""


def render(ch):
    tex = open(os.path.join(SRC, f"chapter{ch}.tex")).read()
    out = os.path.join(BASE, DIRS[ch])
    os.makedirs(out, exist_ok=True)
    n = 0
    for f in re.findall(r"\\begin\{figure\}\[htbp\](.*?)\\end\{figure\}", tex, re.S):
        if "tikzpicture" not in f:
            continue
        lm = re.search(r"\\label\{fig:([A-Za-z0-9]+)\}", f)
        if not lm:
            continue
        name = lm.group(1)
        body = re.sub(r"\\begin\{adjustbox\}.*?\n", "", f)
        body = body.replace("\\end{adjustbox}\n", "").replace("\\centering\n", "")
        body = re.sub(r"\\caption\{.*?\}\s*\\label\{fig:[A-Za-z0-9]+\}", "",
                      body, flags=re.S)
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, name + ".tex")
            open(src, "w").write(HEAD + body + "\n\\end{document}\n")
            subprocess.run(["pdflatex", "-interaction=nonstopmode", src],
                           cwd=tmp, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            pdf = os.path.join(tmp, name + ".pdf")
            if not os.path.exists(pdf):
                print(f"  !! {name}: pdflatex produced no output"); continue
            subprocess.run(["pdftoppm", "-png", "-r", "200", "-singlefile",
                            pdf, os.path.join(out, name)], check=True)
            shutil.copy(pdf, os.path.join(out, name + ".pdf"))
        print(f"  {DIRS[ch]}/{name}.png")
        n += 1
    return n


if __name__ == "__main__":
    chs = [int(a) for a in sys.argv[1:]] or [11]
    print(f"rendered {sum(render(c) for c in chs)} TikZ figures")
