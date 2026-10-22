"""Shared frozen MedGemma loader and bounded input tracing; no experiment imports."""
from brain_vlm.common import require, tensor_description


class TracedGenerator:
    """Observe the first real forward input of each call, without changing it."""
    def __init__(self, adapter):
        self.adapter = adapter
        self.first = False
        self.handle = adapter.model.register_forward_pre_hook(self.capture,with_kwargs=True)
    @property
    def metadata(self):return self.adapter.metadata
    @property
    def model_id(self):return self.adapter.model_id
    def capture(self,model,args,kwargs):
        if not self.first:
            self.adapter.metadata["actual_model_inputs"] = tensor_description(kwargs)
            self.first = True
    def configure_response(self,**kwargs):self.adapter.configure_response(**kwargs)
    def configure_generation(self,**kwargs):self.adapter.configure_generation(**kwargs)
    def set_response_schema(self,schema):self.adapter.set_response_schema(schema)
    def __call__(self,prompt,images):
        self.first = False
        self.adapter.metadata.pop("actual_model_inputs",None)
        return self.adapter(prompt,images)


def load_medgemma(cfg, max_new_tokens):
    """One shared frozen loader for M2 and M7; phases remain separate processes."""
    import torch
    from brain_vlm.baseline.vlm.models import ensure_snapshot
    from brain_vlm.refinement.resources import ResourceMonitor
    from brain_vlm.refinement.local_vlm import LocalVLM, runtime_profile
    require(cfg["device"] != "cuda" or torch.cuda.is_available(),"Select a GPU runtime")
    m = cfg["models"]["assessor"]
    snapshot = ensure_snapshot(m["path"],m["repo_id"],m["revision"],cfg["download_missing"],m.get("config_sha256"))
    props = torch.cuda.get_device_properties(0) if cfg["device"]=="cuda" else None
    profile = runtime_profile("medgemma",props.total_memory/2**30,props.major) if props else {
        "family":"medgemma","dtype":"float32","device_map":"cpu","vision_batch_size":1,"attn_implementation":"sdpa"}
    with ResourceMonitor() as monitor:
        base = LocalVLM.from_local(m["path"],max_new_tokens=max_new_tokens,max_images=cfg["max_images"],**profile)
    return TracedGenerator(base), {"model_loaded":True,"snapshot":snapshot,"resources":monitor.record,"model_metadata":base.metadata.copy()}, {
        "version":"shared_medgemma_v4","snapshot":snapshot,"runtime_profile":profile,"max_new_tokens":max_new_tokens}
