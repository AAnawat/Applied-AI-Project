"""Write private.tex (gitignored) so the next xelatex build fills in the grader login and our Supabase keys.

    python make_private.py <Classroom_Submission.txt> <.streamlit/secrets.toml>

The submission file holds "<key>: <email>" and "password: <password>" lines. Values are never printed.
Every non-alphanumeric character becomes \\char"XX, so passwords with LaTeX specials still typeset.
"""

import re
import sys
import tomllib
from pathlib import Path


def tex(value):
    # \relax ends the hex number, so ")5" doesn't become \char"295
    return "".join(c if c.isascii() and c.isalnum() else f'\\char"{ord(c):X}\\relax ' for c in value)


def main(submission, secrets):
    fields = {}
    for line in Path(submission).read_text(encoding="utf-8-sig").splitlines():
        key, sep, value = line.partition(":")
        if sep and value.strip():
            fields[key.strip().lower()] = value.strip()
    email = next((v for v in fields.values() if re.fullmatch(r"[^@\s]+@[^@\s]+", v)), None)
    password = fields.get("password")

    conf = tomllib.loads(Path(secrets).read_text(encoding="utf-8"))
    url, key = conf.get("supabase_url", ""), conf.get("supabase_key", "")

    checks = {
        "grader email": bool(email),
        "grader password": bool(password),
        "supabase_url looks like https://<ref>.supabase.co": bool(re.fullmatch(r"https://[a-z0-9]+\.supabase\.co/?", url)),
        "supabase_key is a publishable key": key.startswith("sb_publishable_"),
    }
    for name, ok in checks.items():
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
    if not all(checks.values()):
        sys.exit(1)

    out = Path(__file__).with_name("private.tex")
    out.write_text(
        f"\\newcommand{{\\GraderEmail}}{{{tex(email)}}}\n"
        f"\\newcommand{{\\GraderPassword}}{{{tex(password)}}}\n"
        f"\\newcommand{{\\SupaURL}}{{{tex(url.rstrip('/'))}}}\n"
        f"\\newcommand{{\\SupaKey}}{{{tex(key)}}}\n",
        encoding="utf-8",
    )
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
