"""Opt-in VQ forward hooks for visibility. Inputs, outputs and weights unchanged."""
import time


def attach_vq_progress(module, interval_seconds=20):
    """Count actual VQ forwards across cases; no assumed total/ETA or GPU sync."""
    if getattr(module, "_brain_vlm_progress_handles", None):return
    state = {"batches": 0, "patches": 0, "last_print": time.monotonic()}
    def before(model, args):
        if not state["batches"]:
            x = args[0]
            print(f"VQ first input: shape={list(x.shape)}, dtype={x.dtype}, device={x.device}", flush=True)
    def after(model, args, output):
        state["batches"] += 1;state["patches"] += len(args[0])
        now = time.monotonic()
        if state["batches"] == 1 or state["batches"] % 32 == 0 or now-state["last_print"] >= interval_seconds:
            print(f"VQ forwards since model load: {state['batches']} batches / {state['patches']} patches (across cases)", flush=True)
            state["last_print"] = now
        # Explicit None: the hook must not replace the actual output.
        return None
    module._brain_vlm_progress_handles = [module.register_forward_pre_hook(before), module.register_forward_hook(after)]
