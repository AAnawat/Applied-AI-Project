"""Build an allowlisted Colab code bundle; never include credentials/model weights."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from categorization.finetuning.common import ROOT


def package(output):
    repo = ROOT.parent
    output = Path(output)
    if output.exists():
        raise ValueError("Bundle already exists. Use a new --output filename.")
    files = list((ROOT / "core").glob("*.py"))
    allowed_suffixes = {".py", ".md", ".txt", ".ipynb"}
    files += [p for p in (ROOT / "finetuning").iterdir()
              if p.is_file() and p.suffix in allowed_suffixes]
    files += [
        ROOT / "finetuning" / "labelled_template.csv",
        ROOT / "finetuning" / "baseline_audit.json",
        ROOT / "data" / "exams" / "climatelens_105.csv",
        repo / "tests" / "test_finetuning.py",
    ]
    files = sorted(set(files))
    output.parent.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            name = path.relative_to(repo).as_posix()
            if "secrets" in name or path.suffix in {".joblib", ".safetensors", ".bin"}:
                raise ValueError("Unexpected private/model artifact in allowlist.")
            data = path.read_bytes()
            archive.writestr(name, data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        archive.writestr("code_bundle_manifest.json", json.dumps(hashes, indent=2))
    print(f"Created {output} with {len(files)} allowlisted files.")
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/climatelens_finetuning_colab.zip"))
    args = parser.parse_args()
    try:
        package(args.output)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
