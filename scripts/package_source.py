"""Package only source, never recordings, credentials, dependencies, or checkpoints."""
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
destination = root / "dist" / "voicesep-source.zip"
destination.parent.mkdir(exist_ok=True)
files = [root / name for name in ("README.md", "pyproject.toml", "requirements-tested.txt", "LICENSE", "NOTICE", ".gitignore")]
files.extend(root / name for name in ("models/README.md", "reports/README.md", "reports/readiness.json", ".github/workflows/checks.yml"))
for folder in ("src", "configs", "scripts", "notebooks", "docs", "tests", "frontend/src", "frontend/public", "examples"):
    files.extend((root / folder).rglob("*"))
files.extend(root / "frontend" / name for name in ("package.json", "package-lock.json", "index.html", "tsconfig.json", "vite.config.ts"))
with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(set(files)):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pt", ".pth", ".wav", ".flac", ".mp3", ".m4a"}:
            archive.write(path, path.relative_to(root))
print(destination)
