"""Check tracked and unignored files before committing this code-only release."""

from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 50 * 1024 * 1024
TEXT_EXTENSIONS = {".py", ".md", ".txt", ".toml", ".json", ".yaml", ".yml", ".cff"}
SPECIAL_NAMES = {".gitignore", ".gitattributes", "LICENSE", "NOTICE", "Makefile"}
EXCLUDED_DIRECTORIES = {
    "data", "datasets", "outputs", "results", "logs", "runs", "profiles",
    "checkpoints", "hf_cache", ".cache", ".venv", "venv", "__pycache__",
}
PRIVATE_PATH = re.compile(r"/(?:Users|home|blue|orange|scratch)/[A-Za-z0-9_.-]+/")
CREDENTIAL_PATTERNS = (
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"https?://[^\s/@:]+:[^\s/@]+@"),
)


def check_file(root: Path, relative: str) -> list[str]:
    path = root / relative
    parts = Path(relative).parts
    issues = []
    if Path(relative).is_absolute() or ".." in parts:
        return ["path outside repository"]
    if any((root.joinpath(*parts[:i])).is_symlink() for i in range(1, len(parts) + 1)):
        return ["symlink"]
    if not path.exists():
        return []  # A tracked deletion is not part of the next tree.
    if not path.is_file():
        return ["not a regular file"]
    if any(part in EXCLUDED_DIRECTORIES for part in parts[:-1]):
        issues.append("excluded data/cache directory")
    if path.name.startswith(".env") or path.name.lower().startswith("credentials"):
        issues.append("credential filename")
    if path.suffix.lower() not in TEXT_EXTENSIONS and path.name not in SPECIAL_NAMES:
        issues.append("file type outside the code-only release")
    if path.stat().st_size >= MAX_BYTES:
        return issues + ["file is at least 50 MiB"]
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeError, OSError):
        return issues + ["not readable UTF-8 text"]
    if PRIVATE_PATH.search(text):
        issues.append("private filesystem path")
    if any(pattern.search(text) for pattern in CREDENTIAL_PATTERNS):
        issues.append("possible credential (value withheld)")
    return issues


def main() -> int:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        check=True, capture_output=True,
    )
    names = sorted(set(result.stdout.decode().strip("\0").split("\0")) - {""})
    total = 0
    largest = (0, "")
    failed = False
    for name in names:
        issues = check_file(ROOT, name)
        if issues:
            failed = True
            print(f"FAIL {name}: {'; '.join(issues)}")
        path = ROOT / name
        if path.is_file() and not path.is_symlink():
            size = path.stat().st_size
            total += size
            largest = max(largest, (size, name))
    print(f"{len(names)} Git-visible files; {total:,} bytes total.")
    print(f"Largest file: {largest[1]} ({largest[0]:,} bytes).")
    print("Review required." if failed else "Size, file-type, and pattern checks passed.")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
