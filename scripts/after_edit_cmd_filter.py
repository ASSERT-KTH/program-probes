"""Brief AE0: does an edit command also run something beyond the edit itself?

classify(cmd) -> {"strict": bool, "runs_project": bool, "reasons": [...], "python_editor": bool}

Heuristic on the bash command string of the edit turn (one command per assistant turn):
  1. heredoc bodies are set aside (`<< 'EOF' ... EOF`), so code written to a file is not parsed as shell;
  2. quoted strings are blanked, so `sed -i 's/a;b/c/'` or `python -c "..."` bodies are not split;
  3. the rest is split into simple commands on && || ; | & newlines $( ( ) and backticks;
  4. each simple command's program (after VAR=x, sudo, env, time, timeout N, nohup, xargs [opts], {, !, if)
     is looked up: interpreters, test runners, build / compile / lint / type-check tools, package managers,
     running a script by path (./x, x.sh, bash x, sh -c) count as running something; editors and viewers
     (sed, perl -i, awk, cat, echo, printf, tee, cp, mv, rm, mkdir, touch, patch, git, grep, head, ...) do not.
strict        any simple command runs something (an edit made by a Python script counts).
runs_project  as strict, except a Python *editor script* does not count: python fed from a heredoc, a -c
              string, or a .py file written by a heredoc in the same command; no -m, no other arguments;
              its code imports only file / text stdlib modules and has no subprocess / os.system / popen /
              exec / runpy / importlib / __import__ / pytest / unittest / doctest / ast.parse / compile /
              py_compile. Such a script reads and rewrites files and cannot run the project's code.
"""
from __future__ import annotations

import re

HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
SPLIT = re.compile(r"&&|\|\||;|\||&|\n|\$\(|`|\(|\)")
PY = re.compile(r"^(python|python[23](\.\d+)?|pypy3?)$")
TEST = {"pytest", "py.test", "tox", "nox", "jest", "mocha", "vitest", "rspec", "phpunit", "ctest", "karma",
        "ava", "tap", "nosetests", "trial", "behave", "cucumber", "playwright", "cypress"}
BUILD = {"make", "cmake", "ninja", "meson", "bazel", "gcc", "g++", "cc", "c++", "clang", "clang++", "javac",
         "mvn", "gradle", "gradlew", "ant", "sbt", "tsc", "rustc", "cargo", "go", "dotnet", "msbuild", "swift",
         "swiftc", "kotlinc", "scalac", "ghc", "stack", "cabal", "mix", "rebar3", "elm", "webpack", "rollup",
         "vite", "esbuild", "babel", "eslint", "tslint", "flake8", "pylint", "mypy", "ruff", "pyflakes",
         "pycodestyle", "black", "isort", "autopep8", "yapf", "gofmt", "goimports", "golint", "staticcheck",
         "prettier", "rubocop", "phpcs", "phpstan", "psalm", "shellcheck", "clang-format", "rustfmt", "pyright",
         "buf", "protoc"}
RUN = {"node", "nodejs", "deno", "bun", "ts-node", "tsx", "java", "kotlin", "scala", "ruby", "irb", "rake",
       "bundle", "rails", "php", "perl", "Rscript", "R", "julia", "lua", "elixir", "erl", "escript", "groovy",
       "coverage", "jupyter", "ipython", "django-admin", "manage.py", "gunicorn", "uvicorn", "flask", "celery",
       "docker", "bash", "sh", "zsh", "dash", "ksh", "source", ".", "eval", "watch", "parallel", "invoke",
       "inv", "fab", "hatch", "pdm", "poetry", "pipenv", "uv", "uvx", "conda", "mamba", "pip", "pip3", "npm",
       "npx", "yarn", "pnpm", "composer", "gem", "sphinx-build", "asv", "hypothesis", "curl", "wget"}
PREFIX = {"sudo", "env", "time", "nohup", "nice", "stdbuf", "command", "builtin", "then", "do", "else", "{", "!",
          "if", "while", "until", "elif", "exec", "}", "fi", "done"}
SAFE_IMPORTS = {"re", "os", "sys", "json", "pathlib", "textwrap", "fileinput", "io", "shutil", "glob", "difflib",
                "collections", "typing", "string", "itertools", "codecs", "tempfile", "fnmatch", "functools",
                "ast"}
FORBIDDEN_NAMES = {"subprocess", "system", "popen", "Popen", "runpy", "importlib", "__import__", "pytest",
                   "unittest", "doctest", "py_compile", "compileall", "exec", "eval", "execv", "execve", "execvp",
                   "execl", "execlp", "spawnl", "spawnv", "startfile", "check_output", "check_call", "pty"}
PY_C = re.compile(r"python[\d.]*\s+(?:-\w+\s+)*-c\s+(\"(?:\\.|[^\"\\])*\"|'[^']*')")


def _split_heredocs(cmd: str) -> tuple[str, list[tuple[str, str]]]:
    """Shell text without heredoc bodies (operators rewritten as HEREDOCOP), and (operator line, body) pairs."""
    out, bodies, waiting, cur, op_line = [], [], [], [], ""
    for line in cmd.split("\n"):
        if waiting:
            if line.strip() == waiting[0]:
                waiting.pop(0)
                bodies.append((op_line, "\n".join(cur)))
                cur = []
            else:
                cur.append(line)
            continue
        found = [m.group(2) for m in HEREDOC.finditer(line)]
        if found:
            waiting.extend(found)
            op_line = line
            line = HEREDOC.sub(" HEREDOCOP ", line)
        out.append(line)
    if cur:
        bodies.append((op_line, "\n".join(cur)))
    return "\n".join(out), bodies


def _blank_quotes(s: str) -> str:
    s = s.replace("'\\''", "")
    s = re.sub(r"'[^']*'", " Q ", s)
    return re.sub(r'"(?:\\.|[^"\\])*"', " Q ", s)


def _program(words: list[str]) -> tuple[str, list[str]]:
    i = 0
    while i < len(words):
        w = words[i]
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", w) or w in PREFIX:
            i += 1
        elif w == "timeout":
            i += 1
            while i < len(words) and (words[i].startswith("-") or re.match(r"^\d", words[i])):
                i += 1
        elif w == "xargs":
            i += 1
            while i < len(words) and words[i].startswith("-"):
                i += 2 if words[i] in ("-I", "-n", "-P", "-d", "-L", "-s") else 1
        else:
            return w, words[i + 1:]
    return "", []


def _positional(args: list[str]) -> list[str]:
    pos, skip = [], False
    for a in args:
        if skip:
            skip = False
            continue
        if a in (">", "<", ">>", "2>", "&>"):
            skip = True
            continue
        if a.startswith("-") or a == "Q" or re.fullmatch(r"\d+", a):
            continue
        pos.append(a)
    return pos


def _safe_python_code(code: str) -> bool:
    """Code tokens only (strings and comments ignored): stdlib file/text imports, nothing that runs code."""
    import io
    import tokenize
    try:
        toks = [t for t in tokenize.generate_tokens(io.StringIO(code).readline)
                if t.type in (tokenize.NAME, tokenize.OP, tokenize.NEWLINE)]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return False
    for i, t in enumerate(toks):
        prev = toks[i - 1].string if i else ""
        prev2 = toks[i - 2].string if i > 1 else ""
        if t.type != tokenize.NAME:
            continue
        if t.string in FORBIDDEN_NAMES:
            return False
        if t.string == "compile" and not (prev == "." and prev2 == "re"):
            return False
        if t.string == "parse" and prev == "." and prev2 == "ast":
            return False
        if t.string in ("insert", "append") and prev == "." and prev2 == "path":
            return False
        if t.string == "from" and (i == 0 or toks[i - 1].type == tokenize.NEWLINE or prev in (":", ";")):
            if i + 1 >= len(toks) or toks[i + 1].string not in SAFE_IMPORTS:
                return False
        if t.string == "import" and (i == 0 or toks[i - 1].type == tokenize.NEWLINE or prev in (":", ";")):
            j = i + 1
            while j < len(toks) and toks[j].type != tokenize.NEWLINE and toks[j].string != ";":
                if toks[j].type == tokenize.NAME and toks[j - 1].string in ("import", ",") \
                        and toks[j].string not in SAFE_IMPORTS:
                    return False
                j += 1
    return True


def _unescape_dq(s: str) -> str:
    return re.sub(r'\\([\\"$`])', r"\1", s)


def classify(cmd: str) -> dict:
    shell, bodies = _split_heredocs(cmd)
    written = set(re.findall(r">\s*['\"]?([^\s'\"<>;&|]+\.py)", shell))
    reasons: list[str] = []
    py_editor, py_code = True, []
    for line in _blank_quotes(shell).split("\n"):
        for seg in SPLIT.split(line):
            words = seg.replace(">>", " >> ").replace(">", " > ").replace("<", " < ").split()
            prog, args = _program(words)
            if not prog or prog in (">", "<", ">>", "HEREDOCOP", "Q"):
                continue
            base = prog.rsplit("/", 1)[-1]
            if PY.match(base):
                reasons.append("python")
                pos = [a for a in _positional([a for a in args if a not in ("HEREDOCOP", "<")])]
                fed = "-c" in args or "HEREDOCOP" in args
                if "-m" in args or not (fed or (len(pos) == 1 and pos[0].strip("'\"") in written)):
                    py_editor = False
                elif fed and pos:
                    py_editor = False
                elif not fed:
                    tgt = pos[0].strip("'\"")
                    py_code += [b for op, b in bodies if re.search(r">\s*['\"]?" + re.escape(tgt), op)]
            elif base in TEST:
                reasons.append("test")
                py_editor = False
            elif base in BUILD:
                reasons.append("build_lint")
                py_editor = False
            elif base == "perl" and any(a.startswith("-") and "i" in a and not a.startswith("--") for a in args):
                continue  # perl -i / -pi -e: in-place editor
            elif base in RUN or prog.startswith("./") or base.endswith((".sh", ".py", ".js", ".rb", ".pl")):
                reasons.append("run")
                py_editor = False
    # code python reads from a heredoc on its own line, or from -c
    py_code += [b for op, b in bodies if PY.search(op.split("<<")[0].split()[-1].rsplit("/", 1)[-1]
                                                   if op.split("<<")[0].split() else "")
                or re.search(r"\bpython[\d.]*\b[^|;&]*<<", op)]
    for m in PY_C.finditer(shell):
        q = m.group(1)
        py_code.append(_unescape_dq(q[1:-1]) if q[0] == '"' else q[1:-1])
    strict = bool(reasons)
    editor = strict and set(reasons) == {"python"} and py_editor and bool(py_code) \
        and all(_safe_python_code(c) for c in py_code)
    return {"strict": strict, "runs_project": strict and not editor, "reasons": sorted(set(reasons)),
            "python_editor": editor}


if __name__ == "__main__":
    tests = {  # cmd: (strict, runs_project)
        "sed -i 's/a;b/c/' x.py": (False, False),
        "sed -i 's/msg = '\\''SQLite 3.8'\\''/x/' /t/tests.py": (False, False),
        "sed -i 's/x/y/' a.py && python -m pytest tests/test_a.py": (True, True),
        "cat > /testbed/a.py << 'EOF'\nimport os; os.system('x') | y\nEOF": (False, False),
        "cat > /testbed/a.py << 'EOF'\nx=1\nEOF\npython /testbed/a.py": (True, False),
        "cat > /tmp/r.py << 'EOF'\nfrom astropy import units\nEOF\npython /tmp/r.py": (True, True),
        "python3 - << 'EOF'\nimport re\nopen('a').write('b')\nEOF": (True, False),
        "cd /testbed && python << 'EOF'\nwith open('a') as f: c=f.read()\nEOF": (True, False),
        "python3 << 'EOF'\nimport subprocess\nsubprocess.run(['pytest'])\nEOF": (True, True),
        "python3 << 'EOF'\nimport ast\nast.parse(open('a').read())\nEOF": (True, True),
        "python -c \"import re; p='a.py'; open(p,'w').write('x')\"": (True, False),
        "python -c \"import django; print(1)\"": (True, True),
        "python reproduce.py": (True, True),
        "cd /app && npm test": (True, True),
        "cd /app && sed -i 's/a/b/' x.go && go build ./...": (True, True),
        "perl -pi -e 's/a/b/' x.pl": (False, False),
        "perl script.pl": (True, True),
        "sed -i 's/a/b/' x && git diff": (False, False),
        "sed -i 's/a/b/' x | head -5; cat x": (False, False),
        "timeout 60 python reproduce.py 2>&1": (True, True),
        "./run_tests.sh": (True, True),
        "echo 'x' > f && tox -e py": (True, True),
        "find . -name '*.py' | xargs sed -i 's/a/b/'": (False, False),
        "find . -name '*.py' | xargs -I {} python {}": (True, True),
        "printf 'x' >> a.txt": (False, False),
        "bash -n script.sh": (True, True),
        "cat > /tmp/f.py << 'EOF'\nwith open('/t/a.py') as f: c=f.read()\nEOF\npython3 /tmp/f.py && python -c 'import a'": (True, True),
        "cat > /tmp/f.py << 'EOF'\nwith open('/t/a.py') as f: c=f.read()\nEOF\npython3 /tmp/f.py && rm /tmp/f.py": (True, False),
    }
    bad = []
    for c, want in tests.items():
        got = classify(c)
        if (got["strict"], got["runs_project"]) != want:
            bad.append((c, want, got))
    for b in bad:
        print("FAIL", b)
    assert not bad
    print("ok")
