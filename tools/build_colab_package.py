"""Package source only and pin the single notebook to the exact archive bytes."""
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "releases/baseline_01"
PACKAGE = RELEASE / "brain_mri_baseline_clean_v1.zip"
NOTEBOOK = ROOT / "notebooks/Brain_MRI_Baseline_Colab.ipynb"


def main():
    sources = sorted((ROOT / "src/brain_vlm").rglob("*.py")) + [ROOT / "src/requirements.txt"]
    files = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in sources}
    manifest = {"version": "brain_mri_baseline_clean_v1", "weights_included": False,
                "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
    RELEASE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(PACKAGE, "w") as archive:
        for name, data in {**files, "manifest.json": json.dumps(manifest, indent=2).encode()}.items():
            entry = zipfile.ZipInfo(name, (2026, 10, 22, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data)
    digest = hashlib.sha256(PACKAGE.read_bytes()).hexdigest()
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    pins = 0
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        text, count = re.subn(r'(unpack\(PACKAGE_DIR/"brain_mri_baseline_clean_v1.zip", ")[a-f0-9]{64}("\))',
                              lambda m: m[1] + digest + m[2], text)
        pins += count
        cell["source"] = text.splitlines(keepends=True)
    if pins != 1:
        raise ValueError("Expected exactly one source hash pin in the notebook")
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+"\n", encoding="utf-8")
    metadata = {"file": PACKAGE.name, "sha256": digest, "bytes": PACKAGE.stat().st_size,
                "package_modules": len(sources)-1, "weights_included": False}
    (RELEASE / "package.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    readme = ROOT / "README.md"
    text = re.sub(r'(Source ZIP SHA256: `)[a-f0-9]{64}(`)', lambda m: m[1]+digest+m[2],
                  readme.read_text(encoding="utf-8"))
    readme.write_text(text, encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
