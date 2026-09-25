"""Citation-Resolves, secrets and PHI gate -> reports/citation_check.json (exit 1 on any failure).

  * every backticked repo path cited in docs/*.md and README.md exists
  * every run_id (32-hex) / span_id (16-hex) cited in docs/ resolves to a committed trace or log record
  * every "<file>.jsonl line N" citation points at an existing line
  * no secrets anywhere in the repo (Google API keys, private keys, generic key assignments)
  * no plaintext synthetic PHI (patient ids, MRNs, names, phones, emails) in logs/, traces/, reports/
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.guardrails.phi import find_plaintext_phi  # noqa: E402

DOCS = sorted((ROOT / "docs").glob("*.md")) + [ROOT / "README.md"]
EVIDENCE_DIRS = ["logs", "traces", "reports"]
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "chroma", "state", ".phoenix"}
PATH_RE = re.compile(r"`((?:src|mcp_server|scripts|tests|logs|traces|reports|docs|data|config)/[^`\s]*|"
                     r"README\.md|SPEC\.md|requirements\.txt|\.env\.example|\.gitignore|pytest\.ini)`")
RUN_RE = re.compile(r"run_id[`*:\s=]*`?([0-9a-f]{32})\b")
SPAN_RE = re.compile(r"span_id[`*:\s=]*`?([0-9a-f]{16})\b")
LINE_RE = re.compile(r"`?((?:logs|traces|reports)/[\w./-]+\.jsonl)`?\s+line\s+(\d+)")
SECRET_RES = {
    "google_api_key": re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    "private_key": re.compile(r"-----BEGIN (RSA |EC )?PRIVATE KEY-----"),
    "assigned_key": re.compile(r"(?i)(api[_-]?key|secret|token)\s*=\s*['\"][A-Za-z0-9_\-]{24,}['\"]"),
}


def repo_files() -> list[Path]:
    try:
        out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.splitlines()
        return [ROOT / f for f in out if (ROOT / f).is_file()]
    except Exception:
        return [p for p in ROOT.rglob("*") if p.is_file() and not (set(p.relative_to(ROOT).parts) & SKIP_DIRS)]


def known_ids() -> tuple[set[str], set[str]]:
    runs, spans = set(), set()
    files = list((ROOT / "traces").rglob("*.jsonl")) + list((ROOT / "logs").glob("*.jsonl"))
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        runs |= set(re.findall(r'"(?:context\.trace_id|run_id)":\s*"([0-9a-f]{32})"', text))
        spans |= set(re.findall(r'"(?:context\.span_id|span_id|parent_id)":\s*"([0-9a-f]{16})"', text))
    return runs, spans


def main() -> int:
    problems: list[dict] = []
    runs, spans = known_ids()
    checked = {"paths": 0, "run_ids": 0, "span_ids": 0, "line_refs": 0}
    for doc in DOCS:
        if not doc.exists():
            continue
        text = doc.read_text(encoding="utf-8")
        rel = doc.relative_to(ROOT).as_posix()
        for m in PATH_RE.finditer(text):
            target = re.split(r"[:#]", m.group(1))[0].rstrip("/.,)")
            if "*" in target or "<" in target:
                continue
            checked["paths"] += 1
            if not (ROOT / target).exists():
                problems.append({"doc": rel, "type": "missing_path", "ref": target})
        for m in RUN_RE.finditer(text):
            checked["run_ids"] += 1
            if m.group(1) not in runs:
                problems.append({"doc": rel, "type": "unresolved_run_id", "ref": m.group(1)})
        for m in SPAN_RE.finditer(text):
            checked["span_ids"] += 1
            if m.group(1) not in spans:
                problems.append({"doc": rel, "type": "unresolved_span_id", "ref": m.group(1)})
        for m in LINE_RE.finditer(text):
            checked["line_refs"] += 1
            f = ROOT / m.group(1)
            n = int(m.group(2))
            if not f.exists() or n > len(f.read_text(encoding="utf-8").splitlines()):
                problems.append({"doc": rel, "type": "bad_line_ref", "ref": f"{m.group(1)} line {n}"})

    for f in repo_files():
        if f.suffix in (".png", ".parquet", ".sqlite", ".pyc") or f.name == ".env":
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        rel = f.relative_to(ROOT).as_posix()
        for name, rx in SECRET_RES.items():
            if rx.search(text):
                problems.append({"file": rel, "type": f"secret:{name}"})
        if rel.split("/")[0] in EVIDENCE_DIRS:
            hits = find_plaintext_phi(text)
            if hits:
                problems.append({"file": rel, "type": "plaintext_phi", "patterns": hits})
    tracked_env = subprocess.run(["git", "ls-files", ".env"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if tracked_env:
        problems.append({"file": ".env", "type": "secret:env_file_tracked"})

    report = {"producer": "scripts/verify_citations.py", "checked": checked, "known_run_ids": len(runs),
              "known_span_ids": len(spans), "problems": problems, "ok": not problems}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "citation_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"checked": checked, "problems": len(problems)}))
    for p in problems:
        print("  PROBLEM:", p)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
