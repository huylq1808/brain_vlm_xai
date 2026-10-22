"""Reuse -> download missing -> PRIMA extraction. All actions are explicit.

Existing weights/configs are never replaced. Model workers call these functions
in their own environments; importing this file needs only the standard library.
"""
import ast
import importlib.util
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

from brain_vlm.common import read, write, sha, require

MEDGEMMA_REVISION = "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b"


def snapshot_files(root):
    """Check shard index, tokenizer and vision processor; don't load tensors."""
    root = Path(root)
    require((root / "config.json").is_file(), "Missing model config")
    require((root / "tokenizer_config.json").is_file() and
            any((root / n).is_file() for n in ["tokenizer.json", "tokenizer.model", "vocab.json"]), "Missing tokenizer")
    require(any((root / n).is_file() for n in ["preprocessor_config.json", "processor_config.json"]), "Missing vision processor")
    if read(root / "config.json").get("model_type") in {"gemma3","qwen3_vl"}:
        require(read(root / "tokenizer_config.json").get("chat_template") or
                any((root / n).is_file() for n in ["chat_template.json","chat_template.jinja"]),"Missing chat template")
    indices = list(root.glob("*.safetensors.index.json")) + list(root.glob("pytorch_model*.bin.index.json"))
    names = sorted(set(read(indices[0])["weight_map"].values())) if indices else [p.name for p in sorted(root.glob("*.safetensors"))]
    if not names:names=[p.name for p in sorted(root.glob("pytorch_model*.bin"))]
    require(names, "Missing weight shards")
    for name in names:
        path = (root / name).resolve()
        require(path.is_relative_to(root.resolve()) and path.is_file() and path.stat().st_size > 0, "Missing/empty shard: " + name)
    return names


def ensure_snapshot(root, repo_id="google/medgemma-1.5-4b-it", revision=MEDGEMMA_REVISION,
                    download_missing=True, required_config_sha=None):
    """Generic pinned HF vision-model snapshot; download only absent files.

    HF authentication uses the normal token store/environment. No token enters
    a manifest. Gated access/quota failures propagate as download failures.
    """
    root = Path(root)
    require(isinstance(revision, str) and re.fullmatch(r"[0-9a-fA-F]{40}",revision) is not None, "Use a full immutable HF revision")
    if (root / "config.json").is_file() and required_config_sha:
        require(sha(root / "config.json") == required_config_sha, "Existing model config differs; choose another model directory")
    downloaded = []
    try:
        names = snapshot_files(root)
    except ValueError:
        require(download_missing, "Snapshot incomplete and downloads disabled")
        from huggingface_hub import HfApi, hf_hub_download
        info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
        for item in info.siblings:
            name = item.rfilename
            if "/" in name or not name.endswith((".json", ".model", ".safetensors", ".bin", ".txt", ".jinja")):
                continue
            target = root / name
            if target.exists():
                require(target.is_file() and target.stat().st_size > 0 and
                        (item.size is None or target.stat().st_size == item.size), "Existing file differs; refusing overwrite: " + name)
                continue
            cached = Path(hf_hub_download(repo_id, name, revision=revision))
            root.mkdir(parents=True, exist_ok=True)
            part = root / (name + ".partial")
            require(not part.exists(), "Unfinished download copy exists: " + str(part))
            shutil.copyfile(cached, part)
            require(item.size is None or part.stat().st_size == item.size, "Downloaded size mismatch")
            part.replace(target)
            downloaded.append(name)
        names = snapshot_files(root)
    if required_config_sha:
        require(sha(root / "config.json") == required_config_sha, "Downloaded config differs from pinned project model")
    return {"kind":"hf_snapshot", "repo_id":repo_id, "revision":revision, "path":str(root),
            "status":"downloaded_missing" if downloaded else "reused", "downloaded_files":downloaded,
            "config_sha256":sha(root / "config.json"), "weight_files":names,
            "weight_file_bytes":sum((root / n).stat().st_size for n in names),
            "validation":"config/shard inventory and sizes; existing full shard hashes not recomputed"}


def compact_manifest(root):
    """Validate an export without importing Torch or deserializing full PRIMA."""
    root = Path(root)
    m = read(root / "manifest.json")
    require(m["format"] == "prima_encoders_v1" and m["validation"]["status"] == "passed", "Invalid compact PRIMA export")
    require(m["validation"].get("checks"),"Missing completed PRIMA parity checks")
    require({"visual.pt","vq_encoder.pt","text.pt","text_tokenizer/tokenizer_config.json"} <= set(m["files"]),"Incomplete compact component inventory")
    require(any(n in m["files"] for n in ["text_tokenizer/tokenizer.json","text_tokenizer/vocab.json"]),"Missing compact text vocabulary")
    for name, info in m["files"].items():
        p = (root / name).resolve()
        require(p.is_relative_to(root.resolve()) and p.is_file() and p.stat().st_size == info["bytes"] and sha(p) == info["sha256"],
                "Compact PRIMA file changed: " + name)
    return m


def ensure_prima(model_root, compact_root, repo_root, download_missing=True):
    """Use validated compact weights, otherwise fetch missing official sources
    and run the existing parity-checking exporter in a child process.
    """
    model_root, compact_root, repo_root = map(Path, [model_root, compact_root, repo_root])
    if (compact_root / "manifest.json").exists():
        m = compact_manifest(compact_root)
        return {"kind":"prima_compact", "status":"reused", "path":str(compact_root),
                "manifest_sha256":sha(compact_root / "manifest.json"), "file_bytes":m["total_file_bytes"], "parity":m["validation"]}
    require(not compact_root.exists(), "Incomplete compact export exists; preserve it and select a new destination")
    from brain_vlm.prima import runtime
    runtime.clone_prima(repo_root)
    runtime.install_dependencies(repo_root, repo_root.parent / "prima_export_constraints.txt")
    script = repo_root / "tools/download_models_and_setup_test.py"
    # Read constants from the pinned official source. No untrusted pickle is used here.
    constants = {}
    for node in ast.parse(script.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {"FULL_MODEL_ID", "TOKENIZER_ID"} for t in node.targets):
            constants[node.targets[0].id] = ast.literal_eval(node.value)
    trained = model_root / "test/trained_models"
    paths = [("FULL_MODEL_ID", trained / "primafullmodel107.pt"),
             ("TOKENIZER_ID", trained / "tokenizer_model/vqvae_model_step16799.pth")]
    downloaded = []
    for key, target in paths:
        if target.exists():
            require(target.is_file() and zipfile.is_zipfile(target), "Existing source checkpoint is incomplete; refusing overwrite")
            continue
        require(download_missing, "PRIMA source missing and downloads disabled: " + str(target))
        import gdown
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".partial")
        require(not partial.exists(), "PRIMA partial download exists; inspect it before retry")
        gdown.download(id=constants[key], output=str(partial), quiet=False, fuzzy=True)
        require(partial.is_file() and zipfile.is_zipfile(partial), "PRIMA download incomplete/quota failure")
        partial.replace(target)
        downloaded.append(str(target))
    # Official helper overwrites configs. Run it in staging, copy only missing configs.
    staging = model_root / "config_setup"
    staging.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable,str(script),"--repo-root",str(staging),"--skip-download"],check=True)
    for relative in ["prima_config.json", "tokenizer_model/config.json"]:
        target = trained / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(staging / "test/trained_models" / relative, target)
    resources_path=model_root / "config_setup/unified_export_resources.json"
    subprocess.run([sys.executable,"-m","brain_vlm.baseline.vlm.export_worker","--prima-root",str(repo_root),"--model-root",str(model_root),
                    "--destination",str(compact_root),"--resources",str(resources_path)],check=True)
    m = compact_manifest(compact_root)
    return {"kind":"prima_compact", "status":"exported", "path":str(compact_root), "downloaded_sources":downloaded,
            "manifest_sha256":sha(compact_root / "manifest.json"), "file_bytes":m["total_file_bytes"], "parity":m["validation"],
            "export_resources":read(resources_path)}
