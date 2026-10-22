"""Small manifests and model I/O descriptions; no model imports."""
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        # Bounded buffer also works beside model imports on low-RAM workers.
        for part in iter(lambda: stream.read(256 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_torch_save(path, value):
    """Commit a small checkpoint without importing an MRI/model backend."""
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def tensor_description(value):
    """Record actual runtime tensors; bounded values, no model parameters."""
    if isinstance(value, dict):
        return {k: tensor_description(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_description(v) for v in value]
    if hasattr(value, "detach"):
        t = value.detach()
        return {"shape": list(t.shape), "dtype": str(t.dtype), "runtime_device": str(value.device),
                "sample_values": t.reshape(-1)[:8].cpu().tolist(), "sample_is_truncated": t.numel() > 8}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"python_type": type(value).__name__}


def inventory(folder):
    root = Path(folder)
    return {p.relative_to(root).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p)}
            for p in sorted(root.rglob("*")) if p.is_file()}


def verify_inventory(folder, files):
    root = Path(folder).resolve()
    for name, info in files.items():
        p = (root / name).resolve()
        require(p.is_relative_to(root) and p.is_file() and p.stat().st_size == info["bytes"] and sha(p) == info["sha256"],
                "Changed stage artifact: " + name)


def package_root():
    return Path(__file__).resolve().parent


def source_root():
    return package_root().parent


def runtime_sources():
    """Conservative internal dependency closure of the public runtime CLI.

    Unrelated optimize experiments must not invalidate a completed inference.
    Explicit/dynamic internal imports and their parent package files are tracked.
    """
    import ast
    import re
    root=package_root();pending=['brain_vlm.__main__'];seen={}
    def locate(module):
        if module=='brain_vlm':return root/'__init__.py'
        if not module.startswith('brain_vlm.'):return None
        relative=module[len('brain_vlm.'):].replace('.','/')
        for path in [root/(relative+'.py'),root/relative/'__init__.py']:
            if path.is_file():return path
        return None
    while pending:
        module=pending.pop();path=locate(module)
        if path is None or path in seen:continue
        seen[path]=module
        parent=module.rsplit('.',1)[0]
        if parent.startswith('brain_vlm'):pending.append(parent)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            if isinstance(node,ast.Import):pending.extend(a.name for a in node.names if a.name.startswith('brain_vlm'))
            elif isinstance(node,ast.ImportFrom) and node.module and node.module.startswith('brain_vlm'):
                pending.append(node.module)
                pending.extend(node.module+'.'+a.name for a in node.names if a.name!='*')
            elif isinstance(node,ast.Constant) and isinstance(node.value,str):
                for name in re.findall(r'\bbrain_vlm(?:\.[A-Za-z_]\w*)+',node.value):
                    # A string can name a function/class in a subprocess script.
                    while name.startswith('brain_vlm.') and locate(name) is None:name=name.rsplit('.',1)[0]
                    pending.append(name)
    return sorted(seen)
