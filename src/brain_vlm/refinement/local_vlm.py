"""Optional local Hugging Face adapter. No dependency or weight downloads on import."""

from pathlib import Path
import platform
import time

from brain_vlm.refinement.contracts import require

ADAPTER_VERSION = "local_vlm_v9"
JSON_SYSTEM_INSTRUCTION = "Return the requested final JSON only. No thinking block, prose or Markdown."


def generation_stop_ids(model, tokenizer, family=None):
    """Preserve snapshot stop IDs and include MedGemma's actual turn terminator."""
    native = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    ids = [] if native is None else [native] if type(native) is int else native
    require(isinstance(ids, (list, tuple)) and all(type(i) is int and i >= 0 for i in ids),
            "Invalid snapshot EOS token IDs")
    require(type(tokenizer.eos_token_id) is int and tokenizer.eos_token_id >= 0,
            "Generation needs a tokenizer EOS token")
    ids = list(ids) + [tokenizer.eos_token_id]
    if family == "medgemma":
        turn = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        require(type(turn) is int and turn >= 0 and
                tokenizer.convert_ids_to_tokens(turn) == "<end_of_turn>",
                "MedGemma tokenizer is missing its turn terminator")
        ids.append(turn)  # Read the checkpoint tokenizer; never hardcode its numeric ID.
    return list(dict.fromkeys(ids))


def decode_completion(processor, tokens, stop_ids):
    """Decode content without the observed final stop token, not by guessing JSON fields."""
    last = int(tokens[-1]) if len(tokens) else None
    stop = last if last in stop_ids else None
    content = tokens[:-1] if stop is not None else tokens
    return processor.decode(content, skip_special_tokens=True), stop


def runtime_profile(family, total_memory_gib, capability_major):
    """Explicit inference settings; BF16 emulation is not native BF16 support."""
    require(family in {"medgemma", "qwen3_vl"}, "Unsupported model family")
    native_bf16 = capability_major >= 8
    return {"dtype": "bfloat16" if family == "medgemma" or native_bf16 else "float16",
            "quantization": "nf4" if family != "medgemma" and total_memory_gib <= 20 else None,
            "compute_dtype": "bfloat16" if native_bf16 else "float32",
            "vision_batch_size": 1 if family == "medgemma" else None,
            "attn_implementation": "sdpa", "device_map": {"": 0} if family == "medgemma" else "auto"}


def chunk_image_features(pixel_values, encode, batch_size, concatenate):
    """Keep all images and their order while limiting the vision encoder's batch."""
    count = len(pixel_values)
    require(type(batch_size) is int and batch_size > 0, "Invalid vision batch size")
    require(count > 0, "Empty vision input")
    return concatenate([encode(pixel_values[start:start + batch_size])
                        for start in range(0, count, batch_size)], dim=0)


class LocalVLM:
    """Use already-loaded pretrained model/processor objects, or an explicit local snapshot."""

    def __init__(self, model, processor, model_id, max_new_tokens=2048, max_images=16):
        require(type(max_new_tokens) is int and max_new_tokens > 0, "Invalid token budget")
        require(type(max_images) is int and max_images > 0, "Invalid image budget")
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.processor = processor
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.max_images = max_images
        self.response_schema = None
        self.constrained_decoding = True
        self.system_instruction = JSON_SYSTEM_INSTRUCTION
        self.generation_options = {}  # M7 defaults remain identical unless explicitly configured.
        self._tokenizer_data = None
        self.metadata = {"model_id": model_id, "max_new_tokens": max_new_tokens,
                         "max_images": max_images, "do_sample": False,
                         "python": platform.python_version(), "weights_downloaded_by_adapter": False,
                         "adapter_version": ADAPTER_VERSION}

    @classmethod
    def from_local(cls, snapshot_path, family="medgemma", dtype="bfloat16",
                   device_map="auto", max_new_tokens=2048, max_images=16,
                   quantization=None, compute_dtype="float32", vision_batch_size=1,
                   attn_implementation="sdpa"):
        """Requires an existing snapshot; never fetches missing files from the network."""
        path = Path(snapshot_path).resolve()
        require((path / "config.json").is_file(), "Supply a complete local Hugging Face model snapshot")
        require(family in {"medgemma", "qwen3_vl"}, "Unsupported model family")
        import torch
        import transformers
        from transformers import AutoProcessor

        require(dtype in {"bfloat16", "float16", "float32"}, "Unsupported dtype")
        require(compute_dtype in {"bfloat16", "float16", "float32"}, "Unsupported compute dtype")
        require(quantization in {None, "nf4"}, "Unsupported quantization")
        class_name = "Gemma3ForConditionalGeneration" if family == "medgemma" else "Qwen3VLForConditionalGeneration"
        model_class = getattr(transformers, class_name, None)
        require(model_class is not None,
                f"{class_name} unavailable. Use a separate compatible assessor runtime, not PRIMA's pin.")
        processor = AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                                 use_fast=False)
        load_options = dict(dtype=getattr(torch, dtype), device_map=device_map,
                            local_files_only=True, trust_remote_code=False,
                            attn_implementation=attn_implementation)
        if quantization == "nf4":
            # Quantize language-model Linear layers at load time; reuse the official snapshot.
            load_options["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=getattr(torch, compute_dtype),
                llm_int8_skip_modules=["vision_tower", "multi_modal_projector"])
        model = model_class.from_pretrained(path, **load_options)
        adapter = cls(model, processor, str(path), max_new_tokens, max_images)
        if family == "medgemma" and vision_batch_size is not None:
            core = model.model
            require(callable(getattr(core, "get_image_features", None)),
                    "Gemma3 image-feature API changed; use the notebook's pinned Transformers version")
            original_encode = core.get_image_features

            def encode_in_chunks(pixel_values):
                adapter.metadata["last_vision_batches"] = (
                    len(pixel_values) + vision_batch_size - 1) // vision_batch_size
                return chunk_image_features(pixel_values, original_encode, vision_batch_size, torch.cat)

            # Wrap this model instance only. Official PRIMA and Transformers source stay intact.
            core.get_image_features = encode_in_chunks
        adapter.metadata.update(transformers=transformers.__version__, torch=torch.__version__,
                                family=family, dtype=dtype, device_map=device_map,
                                quantization=quantization, compute_dtype=compute_dtype,
                                vision_batch_size=vision_batch_size if family == "medgemma" else None,
                                attn_implementation=attn_implementation,
                                model_commit=getattr(model.config, "_commit_hash", None))
        weight_files = sorted([*path.glob("*.safetensors"), *path.glob("pytorch_model*.bin")])
        adapter.metadata.update(parameter_count=sum(p.numel() for p in model.parameters()),
                                parameter_tensor_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
                                snapshot_weight_file_bytes=sum(p.stat().st_size for p in weight_files),
                                snapshot_weight_files=[p.name for p in weight_files])
        adapter.metadata["eos_token_ids"] = generation_stop_ids(model, processor.tokenizer, family)
        return adapter

    def set_response_schema(self, schema):
        self.response_schema = schema

    def configure_generation(self, no_repeat_ngram_size=0, repetition_penalty=1.0):
        """Explicit M2 text controls; never infer image truth or change M7 defaults."""
        require(type(no_repeat_ngram_size) is int and 0 <= no_repeat_ngram_size <= 32,
                "Invalid repeated-ngram size")
        require(type(repetition_penalty) in {int, float} and 1 <= repetition_penalty <= 2,
                "Invalid repetition penalty")
        self.generation_options = {"no_repeat_ngram_size": no_repeat_ngram_size,
                                   "repetition_penalty": float(repetition_penalty)}
        self.metadata["generation_options"] = self.generation_options.copy()

    def configure_response(self, constrained_decoding=True, system_instruction=JSON_SYSTEM_INSTRUCTION):
        """Opt-in ablations; defaults preserve the original JSON-only generation behavior."""
        require(type(constrained_decoding) is bool, "Constrained decoding must be boolean")
        require(system_instruction is None or isinstance(system_instruction, str)
                and system_instruction.strip(), "Invalid system instruction")
        self.constrained_decoding = constrained_decoding
        self.system_instruction = system_instruction
        self.metadata.update(constrained_decoding=constrained_decoding,
                             system_instruction=system_instruction)

    def _generation_constraints(self):
        require(self.response_schema is not None, "Supply the observation response schema before inference")
        tokenizer = self.processor.tokenizer
        stop_ids = generation_stop_ids(self.model, tokenizer, self.metadata.get("family"))
        self.metadata["eos_token_ids"] = stop_ids
        if not self.constrained_decoding:
            self.metadata.update(output_constraint="none", lm_format_enforcer=None,
                                 response_schema=self.response_schema, use_fast=False)
            return {"eos_token_id": stop_ids}  # Both modes use the same complete stopping rule.
        from lmformatenforcer import JsonSchemaParser, CharacterLevelParserConfig
        from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data, build_transformers_prefix_allowed_tokens_fn
        from importlib.metadata import version
        if self._tokenizer_data is None:
            # Cache vocabulary decoding across jobs; each response gets a fresh parser.
            self._tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)
        config = CharacterLevelParserConfig(alphabet=self._tokenizer_data.tokenizer_alphabet,
                                           force_json_field_order=True, max_consecutive_whitespaces=2)
        parser = JsonSchemaParser(self.response_schema, config)
        prefix = build_transformers_prefix_allowed_tokens_fn(self._tokenizer_data, parser)
        # LMFE 0.11.3 resets parser.config while constructing TokenEnforcer.
        # Restore our per-response settings after its tokenizer-alphabet initialization.
        parser.config = config
        self.metadata.update(output_constraint="json_schema", lm_format_enforcer=version("lm-format-enforcer"),
                             response_schema=self.response_schema, use_fast=False)
        return {"prefix_allowed_tokens_fn": prefix, "eos_token_id": stop_ids}

    def __call__(self, prompt, images):
        from brain_vlm.refinement.resources import ResourceMonitor
        monitor = ResourceMonitor()
        self.metadata.update(last_execution="started", last_image_count=len(images),
                             last_evidence_ids=[x["evidence_id"] for x in images],
                             last_input_tokens=None, last_output_tokens=None,
                             last_pixel_values_shape=None, last_vision_batches=0,
                             last_generation_seconds=None, output_hit_token_cap=None, last_resources=None,
                             last_stop_token_id=None, terminal_token_removed=False)
        try:
            with monitor:
                return self._generate(prompt, images)
        except Exception:
            self.metadata["last_execution"] = "failed"
            raise
        finally:
            self.metadata["last_resources"] = monitor.record

    def _generate(self, prompt, images):
        import torch
        from PIL import Image

        require(len(images) <= self.max_images, "Evidence exceeds image budget; no silent truncation")
        constraints = self._generation_constraints()
        content = []
        for item in images:
            content.append({"type": "text", "text":
                            f"Evidence {item['evidence_id']}; sequence={item['sequence']}; scope={item['scope']}; "
                            f"role={item.get('role','native_view')}; native_z={item.get('native_canonical_z','not_recorded')}; "
                            f"paired_context={item.get('paired_context_id','none')}."})
            with Image.open(item["resolved_path"]) as source:
                content.append({"type": "image", "image": source.convert("RGB").copy()})
        content.append({"type": "text", "text": prompt})
        messages = []
        if self.system_instruction is not None:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system_instruction}]})
        messages.append({"role": "user", "content": content})
        processor_options = {"do_pan_and_scan": False} if self.metadata.get("family") == "medgemma" else {}
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt", **processor_options)
        device = self.model.get_input_embeddings().weight.device
        require(device.type != "meta", "Model input device is unresolved")
        inputs = inputs.to(device)
        for key, value in inputs.items():
            if torch.is_floating_point(value):
                inputs[key] = value.to(dtype=self.model.dtype)
        start = time.perf_counter()
        # Update before prefill so an OOM cannot leave metadata from the previous text-only job.
        self.metadata.update(last_image_count=len(images),
                             last_evidence_ids=[x["evidence_id"] for x in images],
                             last_input_tokens=inputs["input_ids"].shape[-1],
                             last_pixel_values_shape=list(inputs["pixel_values"].shape)
                             if "pixel_values" in inputs else None,
                             last_vision_batches=0, last_output_tokens=None,
                             output_hit_token_cap=None, last_generation_seconds=None,
                             last_execution="started")
        try:
            with torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                             **constraints, **self.generation_options)
        except Exception:
            self.metadata.update(last_execution="failed", last_generation_seconds=time.perf_counter() - start)
            raise
        tokens = output[0][inputs["input_ids"].shape[-1]:]
        response, stop_id = decode_completion(self.processor, tokens, constraints["eos_token_id"])
        self.metadata.update(last_image_count=len(images), last_input_tokens=inputs["input_ids"].shape[-1],
                             last_output_tokens=len(tokens), last_generation_seconds=time.perf_counter() - start,
                             output_hit_token_cap=len(tokens) >= self.max_new_tokens, last_execution="generated",
                             last_stop_token_id=stop_id, terminal_token_removed=stop_id is not None)
        return response
