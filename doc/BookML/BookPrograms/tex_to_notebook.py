import re, json, os, sys, ast, uuid

SRC = "/sessions/kind-inspiring-knuth/mnt/MachineLearning/doc/BookML"
CHAPTERS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

# BookFigures subdirectory per chapter, used for TikZ schematics that have no
# \includegraphics and are rendered separately by BookFigures/render_tikz.py.
DIRS = {1:"chapter01_linear_algebra", 2:"chapter02_statistics",
        3:"chapter03_linear_regression", 4:"chapter04_optimization",
        5:"chapter05_logistic_regression", 6:"chapter06_support_vector_machines",
        7:"chapter07_trees_and_ensembles", 8:"chapter08_neural_networks",
        9:"chapter09_differential_equations",
        10:"chapter10_convolutional_networks", 11:"chapter11_recurrent_networks", 12:"chapter12_autoencoders", 13:"chapter13_transformers", 14:"chapter14_boltzmann", 15:"chapter15_vae", 16:"chapter16_diffusion"}

MATH_ENVS = ("equation","equation*","align","align*","eqnarray","eqnarray*")

PREAMBLE = r"""$$
\newcommand{\bm}[1]{\boldsymbol{#1}}
\newcommand{\Det}[1]{|\boldsymbol{#1}|}
\newcommand{\bigO}{\mathcal{O}}
\newcommand{\var}{\mathrm{Var}}
\newcommand{\cov}{\mathrm{Cov}}
\newcommand{\Prob}{\mathrm{Prob}}
\newcommand{\mean}[1]{\langle #1 \rangle}
$$"""

def strip_comments(s):
    """Strip LaTeX %-comments, but never inside a verbatim listing, where a
    bare % is legitimate Python (a format specifier, a modulo, an f-string)."""
    out=[]; in_code=False
    for line in s.split("\n"):
        if re.match(r"\s*\\begin\{(Python|C\+\+)\}", line):
            in_code=True; out.append(line); continue
        if re.match(r"\s*\\end\{(Python|C\+\+)\}", line):
            in_code=False; out.append(line); continue
        if in_code:
            out.append(line); continue
        i=0; res=""
        while i < len(line):
            if line[i]=="%" and (i==0 or line[i-1]!="\\"):
                break
            res+=line[i]; i+=1
        out.append(res)
    return "\n".join(out)

def read(ch):
    return strip_comments(open(f"{SRC}/chapter{ch}.tex").read())

# ---------------------------------------------------------------- pass 1
def build_maps():
    eqnum, secname, chapname, tabnum = {}, {}, {}, {}
    for ch in CHAPTERS:
        s = read(ch)
        m = re.search(r"\\chapter(?:\[[^\]]*\])?\{(.*?)\}\s*\n", s, re.S)
        title = m.group(1).strip() if m else f"Chapter {ch}"
        lm = re.search(r"\\label\{(chap:[^}]*)\}", s)
        if lm: chapname[lm.group(1)] = (ch, title)
        # section / subsection labels -> title
        for sm in re.finditer(r"\\(sub)?section\*?\{(.*?)\}\s*\n\s*\\label\{([^}]*)\}", s, re.S):
            secname[sm.group(3)] = sm.group(2).strip()
        # table numbering
        tn = 0
        for tm2 in re.finditer(r"\\begin\{table\}(.*?)\\end\{table\}", s, re.S):
            tn += 1
            for l in re.findall(r"\\label\{(tab:[^}]*)\}", tm2.group(1)): tabnum[l] = f"{ch}.{tn}"
        # equation numbering, in document order
        n = 0
        for em in re.finditer(r"\\begin\{(equation|align|eqnarray)\}(.*?)\\end\{\1\}", s, re.S):
            body = em.group(2)
            labels = re.findall(r"\\label\{([^}]*)\}", body)
            if em.group(1) in ("align","eqnarray"):
                # one number per \\ separated row that carries a label
                rows = body.split(r"\\")
                for r in rows:
                    ls = re.findall(r"\\label\{([^}]*)\}", r)
                    if "\\nonumber" in r: continue
                    n += 1
                    for l in ls: eqnum[l] = f"{ch}.{n}"
            else:
                n += 1
                for l in labels: eqnum[l] = f"{ch}.{n}"
    return eqnum, secname, chapname, tabnum

EQ, SEC, CHAP, TAB = build_maps()

def resolve_ref(label):
    """Return the bare reference text; the source already supplies
    'Eq.~(...)', 'Chapter~', 'Section~' and so on."""
    if label in EQ:   return EQ[label]
    if label in TAB:  return TAB[label]
    if label in CHAP: return str(CHAP[label][0])
    if label in SEC:  return f"*{SEC[label]}*"
    return label

# ---------------------------------------------------------------- inline
def inline(s):
    _st = []
    def _stash(m):
        _st.append(m.group(0)); return f"\x01M{len(_st)-1}\x01"
    s = re.sub(r"\$\$.*?\$\$", _stash, s, flags=re.S)
    s = re.sub(r"\$(?:[^$\n]|\n(?!\n))*?\$", _stash, s)
    s = re.sub(r"\\(?:Eq|Section|Chapter|Table|Figure)?~?\\?ref\{([^}]*)\}",
               lambda m: resolve_ref(m.group(1)), s)
    s = re.sub(r"\\ref\{([^}]*)\}", lambda m: resolve_ref(m.group(1)), s)
    s = re.sub(r"\\cite\{([^}]*)\}", lambda m: f"[{m.group(1)}]", s)
    s = re.sub(r"\\index\{[^}]*\}", "", s)
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    s = re.sub(r"\\emph\{(.*?)\}", r"*\1*", s, flags=re.S)
    s = re.sub(r"\\textbf\{(.*?)\}", r"**\1**", s, flags=re.S)
    s = re.sub(r"\\textit\{(.*?)\}", r"*\1*", s, flags=re.S)
    s = re.sub(r"\\texttt\{(.*?)\}", lambda m: "`"+m.group(1).replace("\\_","_").replace("\\","")+"`", s, flags=re.S)
    s = re.sub(r"\\href\{([^}]*)\}\{([^}]*)\}", r"[\2](\1)", s)
    s = re.sub(r"\\url\{([^}]*)\}", r"<\1>", s)
    s = re.sub(r"\\footnote\{(.*?)\}", r" (\1)", s, flags=re.S)
    s = s.replace("\\%","%").replace("\\&","&").replace("\\#","#")
    s = re.sub(r"(?<!`)``(?!`)", '"', s)
    s = s.replace("''", '"')
    # text-mode LaTeX that legitimately appears outside math
    for a,b in [(r"\\ldots","..."), (r"\\dots","..."), (r"\\cdots","..."),
                (r"\\times"," x "), (r"\\ ", " "), (r"\\,", " "), (r"\\;", " "),
                (r"\\quad","  "), (r"\\qquad","    "), (r"\\hspace\{[^}]*\}"," "),
                (r"\\vspace\{[^}]*\}",""), (r"\\smallskip",""), (r"\\medskip",""),
                (r"\\bigskip",""), (r"\\/",""), (r"\\@","")]:
        s = re.sub(a, b, s)
    s = re.sub(r"\\-", "", s)
    s = s.replace("~"," ")
    s = re.sub(r"\\\\(?![a-zA-Z])", "  \n", s)
    s = re.sub(r"\x01M(\d+)\x01", lambda m: _st[int(m.group(1))], s)
    return s

def tabular_to_md(body):
    body = re.sub(r"\\(top|mid|bottom)rule","",body)
    body = re.sub(r"\\hline","",body)
    rows=[r.strip() for r in body.split(r"\\") if r.strip()]
    out=[]
    for i,r in enumerate(rows):
        cells=[inline(c.strip()) for c in r.split("&")]
        out.append("| "+" | ".join(cells)+" |")
        if i==0: out.append("|"+"|".join(["---"]*len(cells))+"|")
    return "\n".join(out)

def convert_prose(s, chnum):
    s = re.sub(r"\\index\{(?:[^{}]|\{[^{}]*\})*\}", "", s)

    # figures -> markdown image + italic caption.  The .pdf used by LaTeX has a
    # .png twin written by the same BookFigures script; point at the .png and
    # make the path relative to doc/LectureNotes.
    def fig_repl(m):
        inner = m.group(1)
        gm = re.search(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", inner)
        cm = re.search(r"\\caption\{(.*?)\}\s*(?:\\label|\\end)", inner, re.S)
        if not gm:
            # A TikZ schematic rather than a generated plot.  These are rendered
            # to PNG by BookFigures/render_tikz.py and named after their label.
            if "tikzpicture" in inner:
                lm = re.search(r"\\label\{fig:([A-Za-z0-9]+)\}", inner)
                if lm:
                    cm2 = re.search(r"\\caption\{(.*?)\}\s*\\label", inner, re.S)
                    cap2 = inline(cm2.group(1)).strip().replace("\n", " ") if cm2 else ""
                    alt2 = re.sub(r"[^A-Za-z0-9 ,.-]", "",
                                  re.sub(r"\$[^$]*\$", "", cap2)).strip()[:70]
                    p = f"../BookML/BookFigures/{DIRS[chnum]}/{lm.group(1)}.png"
                    return f"\n\n![{alt2}]({p})\n\n*{cap2}*\n\n"
            return ""
        path = "../BookML/" + gm.group(1) + ".png"
        cap = inline(cm.group(1)).strip().replace("\n", " ") if cm else ""
        alt = re.sub(r"\\$[^$]*\\$", "", cap)          # no math in the alt text
        alt = re.sub(r"[^A-Za-z0-9 ,.-]", "", alt).strip()[:70]
        return f"\n\n![{alt}]({path})\n\n*{cap}*\n\n"
    s = re.sub(r"\\begin\{figure\}\[?[^\]]*\]?(.*?)\\end\{figure\}",
               fig_repl, s, flags=re.S)
    # tables
    def table_repl(m):
        inner=m.group(1)
        tm=re.search(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", inner, re.S)
        cm=re.search(r"\\caption\{(.*?)\}\s*(?:\\label|\Z)", inner, re.S)
        md = tabular_to_md(tm.group(1)) if tm else ""
        cap = ("\n\n*"+inline(cm.group(1)).strip()+"*") if cm else ""
        return "\n\n"+md+cap+"\n\n"
    s = re.sub(r"\\begin\{table\}\[?[^\]]*\]?(.*?)\\end\{table\}", table_repl, s, flags=re.S)
    s = re.sub(r"\\begin\{center\}(.*?)\\end\{center\}", lambda m: table_repl(m), s, flags=re.S)

    # notebox -> admonition
    def note_repl(m):
        inner = convert_prose(m.group(1), chnum).strip()
        title = "Note"
        tm = re.match(r"\*\*(.*?)\*\*\s*", inner)
        if tm:
            title = tm.group(1).rstrip(".")
            inner = inner[tm.end():]
        body = "\n".join("  "+l if l.strip() else "" for l in inner.split("\n"))
        return f"\n\n```{{admonition}} {title}\n:class: tip\n{inner}\n```\n\n"
    s = re.sub(r"\\begin\{notebox\}(.*?)\\end\{notebox\}", note_repl, s, flags=re.S)

    # theorem-like environments -> titled admonitions (Chapter 10 onwards)
    THMENV = {"theorem": "Theorem", "proposition": "Proposition",
              "lemma": "Lemma", "corollary": "Corollary",
              "definition": "Definition", "example": "Example"}
    def thm_repl(env, cls):
        def _f(m):
            opt, body = m.group(1), m.group(2)
            name = re.sub(r"^\[|\]$", "", opt or "").strip()
            title = THMENV[env] + (f" ({name})" if name else "")
            inner = convert_prose(body, chnum).strip()
            return f"\n\n```{{admonition}} {title}\n:class: {cls}\n{inner}\n```\n\n"
        return _f
    for env, cls in [(e, "important") for e in THMENV]:
        s = re.sub(r"\\begin\{" + env + r"\}(\[[^\]]*\])?(.*?)\\end\{" + env + r"\}",
                   thm_repl(env, cls), s, flags=re.S)
    def proof_repl(m):
        inner = convert_prose(m.group(1), chnum).strip()
        return f"\n\n```{{admonition}} Proof\n:class: note\n{inner}\n```\n\n"
    s = re.sub(r"\\begin\{proof\}(.*?)\\end\{proof\}", proof_repl, s, flags=re.S)

    # display math
    def math_repl(m):
        env, body = m.group(1), m.group(2)
        labels = re.findall(r"\\label\{([^}]*)\}", body)
        tag = ""
        if labels and labels[0] in EQ: tag = "\\tag{%s}" % EQ[labels[0]]
        body = re.sub(r"\\label\{[^}]*\}", "", body)
        if env.startswith("equation"):
            return "\n\n$$\n"+body.strip()+tag+"\n$$\n\n"
        return "\n\n$$\n\\begin{"+env+"}\n"+body.strip()+"\n\\end{"+env+"}\n$$\n\n"
    s = re.sub(r"\\begin\{(equation\*?|align\*?|eqnarray\*?)\}(.*?)\\end\{\1\}", math_repl, s, flags=re.S)
    s = re.sub(r"(?<!\\)\\\[(.*?)(?<!\\)\\\]",
               lambda m: "\n\n$$\n"+m.group(1).strip()+"\n$$\n\n", s, flags=re.S)

    # lists, innermost first so that nesting is handled correctly
    def list_repl(body, ordered, indent):
        items = re.split(r"\\item\s", body)[1:]
        pad = "    " * indent
        out = []
        for i, t in enumerate(items):
            t = inline(t).strip()
            marker = f"{i+1}." if ordered else "-"
            lines = t.split("\n")
            first = pad + marker + " " + lines[0].strip()
            rest = [pad + "   " + l.strip() if l.strip() else "" for l in lines[1:]]
            out.append("\n".join([first] + rest))
        return "\n\n" + "\n".join(out) + "\n\n"

    def expand_lists(text, indent=0):
        pat = re.compile(r"\\begin\{(itemize|enumerate)\}"
                         r"((?:(?!\\begin\{(?:itemize|enumerate)\}).)*?)"
                         r"\\end\{\1\}", re.S)
        while True:
            m = pat.search(text)
            if not m: break
            text = text[:m.start()] + list_repl(m.group(2), m.group(1)=="enumerate",
                                                indent) + text[m.end():]
        return text
    s = expand_lists(s)

    # sectioning
    s = re.sub(r"\\section\*?\{(.*?)\}", lambda m: "\n\n## "+inline(m.group(1))+"\n", s, flags=re.S)
    s = re.sub(r"\\subsection\*?\{(.*?)\}", lambda m: "\n\n### "+inline(m.group(1))+"\n", s, flags=re.S)
    s = re.sub(r"\\paragraph\{(.*?)\}", lambda m: "\n\n**"+inline(m.group(1)).rstrip(".")+".** ", s, flags=re.S)
    s = re.sub(r"\\addcontentsline\{[^}]*\}\{[^}]*\}\{[^}]*\}", "", s)
    s = re.sub(r"\\chaptermark\{[^}]*\}", "", s)
    s = re.sub(r"\\(clearemptydoublepage|newpage|noindent|centering|toprule|midrule|bottomrule)\b", "", s)
    s = re.sub(r"\\renewcommand\{[^}]*\}\{[^}]*\}", "", s)
    s = re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}", "", s)

    # protect math so that the text-mode substitutions in inline() cannot
    # reach inside it
    store = []
    def stash(m):
        store.append(m.group(0)); return f"\x00MATH{len(store)-1}\x00"
    s = re.sub(r"\$\$.*?\$\$", stash, s, flags=re.S)
    s = re.sub(r"\$(?:[^$\n]|\n(?!\n))*?\$", stash, s)

    s = inline(s)

    s = re.sub(r"\x00MATH(\d+)\x00", lambda m: store[int(m.group(1))], s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def convert_chapter(ch):
    s = read(ch)
    m = re.search(r"\\chapter(?:\[[^\]]*\])?\{(.*?)\}\s*\n", s, re.S)
    title = m.group(1).strip()
    s = s[m.end():]
    s = re.sub(r"^\s*\\chaptermark\{[^}]*\}\s*\n","",s)
    s = re.sub(r"^\s*\\label\{[^}]*\}\s*\n","",s)

    cells=[{"cell_type":"markdown","metadata":{},
            "source":[f"# Chapter {ch}: {title}\n"]},
           {"cell_type":"markdown","metadata":{},
            "source":["<!-- Macro definitions for MathJax, mirroring book.tex -->\n", PREAMBLE]}]

    parts = re.split(r"\\begin\{Python\}\{\}\n(.*?)\\end\{Python\}", s, flags=re.S)
    for i,part in enumerate(parts):
        if i % 2 == 0:
            md = convert_prose(part, ch)
            if md:
                # split into one cell per section/subsection so the notebook
                # stays navigable rather than one enormous cell
                chunks, cur = [], []
                for line in md.split("\n"):
                    if line.startswith("## ") and cur and any(x.strip() for x in cur):
                        chunks.append("\n".join(cur).strip()); cur=[line]
                    else:
                        cur.append(line)
                if cur: chunks.append("\n".join(cur).strip())
                for chunk in chunks:
                    if chunk:
                        cells.append({"cell_type":"markdown","metadata":{},
                                      "source":[l+"\n" for l in chunk.split("\n")]})
        else:
            code = part.rstrip("\n")
            try:
                ast.parse(code); is_output = False
            except SyntaxError:
                is_output = True          # captured program output, not source
            if is_output:
                cells.append({"cell_type":"markdown","metadata":{},
                              "source":["```\n"]+[l+"\n" for l in code.split("\n")]+["```\n"]})
            else:
                cells.append({"cell_type":"code","execution_count":None,"metadata":{},
                              "outputs":[], "source":[l+"\n" for l in code.split("\n")]})
    for c in cells: c["id"] = uuid.uuid4().hex[:12]
    nb={"cells":cells,
        "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                    "language_info":{"name":"python","version":"3.11"}},
        "nbformat":4,"nbformat_minor":5}
    return nb

if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for ch in CHAPTERS:
        nb = convert_chapter(ch)
        with open(f"{out}/chapter{ch}.ipynb","w") as f: json.dump(nb,f,indent=1)
        nmd=sum(1 for c in nb["cells"] if c["cell_type"]=="markdown")
        ncode=sum(1 for c in nb["cells"] if c["cell_type"]=="code")
        print(f"chapter{ch}.ipynb: {nmd} markdown, {ncode} code cells")
