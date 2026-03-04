"""
CoddLlada — LLaDAModelLM extended with PyJuice PC guidance.

Inherits all LLaDA functionality and adds:
- PC logit modification in ``forward()``
- Unified ``save_pretrained()`` / ``from_pretrained()`` for base model + PC
- ``push_to_hub()`` for easy sharing
- ``generate()`` convenience method for LLaDA diffusion generation
"""

import os
import sys
import json

import torch
import torch.nn.functional as F
import pyjuice as juice

from typing import Optional, Tuple, Union, List
from dataclasses import fields, asdict
from transformers import AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from huggingface_hub import hf_hub_download

from .modeling_llada import LLaDAModelLM
from .configuration_llada import LLaDAConfig

# Ensure the project root is on sys.path so `codd` package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from codd.model_class import CoDDConfig, CoDDOutput, apply_pc_logits, PyJuiceHubModel


class CoddLlada(LLaDAModelLM):
    """LLaDA model with integrated PyJuice Probabilistic Circuit (PC).

    This is a *proper subclass* of :class:`LLaDAModelLM` — it IS a
    ``PreTrainedModel`` with all the standard HuggingFace API surface
    (``save_pretrained``, ``from_pretrained``, ``push_to_hub``, …).

    The PC model is stored as a submodule but excluded from the regular
    state-dict so that normal HF weight saving/loading is unaffected;
    the PC is persisted separately as ``pc_model.jpc`` via PyJuice, and
    CoDD-specific metadata is saved as ``codd_config.json``.

    Usage::

        # ── From separate repos ──
        model = CoddLlada.from_pretrained(
            "GSAI-ML/LLaDA-8B-Instruct",
            pc_model_id="il18/llada-math-pc",
            torch_dtype=torch.bfloat16,
        )

        # ── From unified repo (created by save_pretrained) ──
        model = CoddLlada.from_pretrained("user/codd-llada-math")

        # ── Save / push ──
        model.save_pretrained("./my-codd-llada")
        model.push_to_hub("user/codd-llada-math")
    """

    config_class = LLaDAConfig
    # Safety: ignore pc_model keys during weight loading in either direction
    _keys_to_ignore_on_load_missing = [r"pc_model\."]
    _keys_to_ignore_on_load_unexpected = [r"pc_model\."]

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def __init__(self, config: LLaDAConfig, model=None, init_params: bool = False):
        super().__init__(config, model=model, init_params=init_params)
        # CoDD-specific state (not registered as HF config)
        object.__setattr__(self, "codd_config", CoDDConfig(base_model_type="llada"))
        object.__setattr__(self, "tokenizer", None)
        self.pc_model = None          # becomes nn.Module submodule when set
        self._pc_compiled = False

    # ------------------------------------------------------------------
    # state_dict — exclude PC weights (saved separately as .jpc)
    # ------------------------------------------------------------------

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        return {k: v for k, v in sd.items() if not k.startswith("pc_model.")}

    # ------------------------------------------------------------------
    # Resolve helper
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_file(path_or_repo: str, filename: str) -> Optional[str]:
        """Check local directory first, fall back to Hub download."""
        local = os.path.join(path_or_repo, filename)
        if os.path.isfile(local):
            return local
        try:
            return hf_hub_download(repo_id=path_or_repo, filename=filename)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # from_pretrained
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        pc_model_id: Optional[str] = None,
        **kwargs,
    ):
        """Load a CoddLlada model.

        Supports two modes, auto-detected:

        1. **Unified repo** — a single path/repo containing the base-model
           weights *and* ``pc_model.jpc`` + ``codd_config.json``.
        2. **Separate repos** — ``pretrained_model_name_or_path`` is the
           base LLaDA model and ``pc_model_id`` points at a separate
           PyJuice Hub repo.

        Any :class:`CoDDConfig` field (``mask_id``, ``vocab_size``, …) may
        be passed as a keyword argument; everything else is forwarded to
        ``LLaDAModelLM.from_pretrained``.
        """
        # --- Separate CoDDConfig fields from HF kwargs ---
        codd_field_names = {f.name for f in fields(CoDDConfig)}
        config_overrides = {
            k: kwargs.pop(k) for k in list(kwargs) if k in codd_field_names
        }

        # --- Detect unified repo ---
        codd_config_path = cls._resolve_file(
            pretrained_model_name_or_path, "codd_config.json"
        )
        if codd_config_path is not None:
            with open(codd_config_path) as f:
                saved = json.load(f)
            saved.update(config_overrides)
            codd_cfg = CoDDConfig(**saved)
        else:
            codd_cfg = CoDDConfig(
                base_model_id=pretrained_model_name_or_path,
                pc_model_id=pc_model_id or "",
                base_model_type="llada",
                **config_overrides,
            )

        # --- Load base LLaDA weights via the standard HF path ---
        instance = super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

        # --- Attach CoDD state ---
        object.__setattr__(instance, "codd_config", codd_cfg)
        instance._pc_compiled = False

        # --- Tokenizer ---
        tk_kwargs = {"trust_remote_code": kwargs.get("trust_remote_code", True)}
        if "local_files_only" in kwargs:
            tk_kwargs["local_files_only"] = kwargs["local_files_only"]
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **tk_kwargs
        )
        object.__setattr__(instance, "tokenizer", tokenizer)

        # --- PC model ---
        instance.pc_model = None
        pc_path = cls._resolve_file(pretrained_model_name_or_path, "pc_model.jpc")
        if pc_path is not None:
            pc_raw = juice.load(pc_path)
            device = next(instance.parameters()).device
            instance.pc_model = juice.compile(pc_raw).to(device)
        elif pc_model_id is not None:
            pc_raw = PyJuiceHubModel.from_pretrained(pc_model_id).pc
            device = next(instance.parameters()).device
            instance.pc_model = juice.compile(pc_raw).to(device)

        return instance

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        # ---- CoDD-specific ----
        pc_block_range: Optional[Tuple[int, int]] = None,
        use_pc: bool = True,
        cfg_scale: float = 0.0,
        prompt_index: Optional[torch.BoolTensor] = None,
        pc_temperature: Optional[float] = None,
        pc_frac: Optional[float] = None,
        reverse_frac: Optional[bool] = None,
        # ---- standard LLaDA kwargs ----
        **kwargs,
    ) -> CoDDOutput:
        """LLaDA forward with optional CFG and PC logit modification.

        When called *without* any CoDD-specific arguments this behaves
        identically to ``LLaDAModelLM.forward()``.
        """
        cfg = self.codd_config

        # --- CFG (classifier-free guidance) ---
        if cfg_scale > 0.0 and prompt_index is not None:
            mask_id = cfg.mask_id
            un_x = input_ids.clone()
            un_x[prompt_index] = mask_id
            x_ = torch.cat([input_ids, un_x], dim=0)
            output = super().forward(x_, **kwargs)
            logits, un_logits = torch.chunk(output.logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            output = super().forward(input_ids, **kwargs)
            logits = output.logits

        # --- PC logit modification ---
        if use_pc and self.pc_model is not None and pc_block_range is not None:
            _pct = pc_temperature if pc_temperature is not None else cfg.pc_temperature
            _pcf = pc_frac if pc_frac is not None else cfg.pc_frac
            _rf = reverse_frac if reverse_frac is not None else cfg.reverse_frac

            record = not self._pc_compiled
            logits = apply_pc_logits(
                pc_model=self.pc_model,
                input_ids=input_ids,
                logits=logits,
                start_idx=pc_block_range[0],
                end_idx=pc_block_range[1],
                mask_id=cfg.mask_id,
                vocab_size=cfg.vocab_size,
                pc_temperature=_pct,
                pc_frac=_pcf,
                reverse_frac=_rf,
                record_cudagraph=record,
            )
            if record:
                self._pc_compiled = True

        return CoDDOutput(logits=logits)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: Union[str, torch.Tensor],
        *,
        generate_fn=None,
        use_pc: bool = True,
        system_instruction: Optional[str] = None,
        num_steps: Optional[int] = None,
        gen_length: Optional[int] = None,
        block_length: Optional[int] = None,
        temperature: Optional[float] = None,
        cfg_scale: Optional[float] = None,
        remasking: Optional[str] = None,
        pc_temperature: Optional[float] = None,
        pc_frac: Optional[float] = None,
        reverse_frac: Optional[bool] = None,
    ) -> torch.Tensor:
        """Run LLaDA diffusion generation with optional PC guidance.

        Args:
            prompt: String or token-id tensor ``(1, L)``.
            generate_fn: Generation function. Auto-imports
                ``llada_diffusion_generate`` when ``None``.
            use_pc: Activate PC-guided generation.
            system_instruction: Optional system prompt (string prompts only).
            Remaining kwargs override :class:`CoDDConfig` defaults.

        Returns:
            Token-id tensor ``(1, prompt_length + gen_length)``.
        """
        if generate_fn is None:
            try:
                from eval.llada.llada_generate import llada_diffusion_generate
            except ImportError:
                from llada.llada_generate import llada_diffusion_generate
            generate_fn = llada_diffusion_generate

        cfg = self.codd_config
        num_steps = num_steps if num_steps is not None else cfg.num_steps
        gen_length = gen_length if gen_length is not None else cfg.gen_length
        block_length = block_length if block_length is not None else cfg.block_length
        temperature = temperature if temperature is not None else cfg.temperature
        cfg_scale = cfg_scale if cfg_scale is not None else cfg.cfg_scale
        remasking = remasking if remasking is not None else cfg.remasking

        if isinstance(prompt, str):
            prompt = self._format_prompt(prompt, system_instruction)

        prompt_ids = prompt.to(self.device)

        return generate_fn(
            model=self,
            prompt=prompt_ids,
            num_steps=num_steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking=remasking,
            mask_id=cfg.mask_id,
            use_pc=use_pc,
            pc_temperature=pc_temperature,
            pc_frac=pc_frac,
            reverse_frac=reverse_frac,
        )

    # ------------------------------------------------------------------
    # Save / Push
    # ------------------------------------------------------------------

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save the unified CoddLlada model.

        Writes:
        - Base model weights + config (via ``LLaDAModelLM.save_pretrained``)
        - Tokenizer files
        - ``pc_model.jpc`` — the PC (via ``juice.save``)
        - ``codd_config.json`` — CoDD metadata
        """
        os.makedirs(save_directory, exist_ok=True)

        # Base model (state_dict override excludes PC weights)
        super().save_pretrained(save_directory, **kwargs)

        # Tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory)

        # PC
        if self.pc_model is not None:
            juice.save(os.path.join(save_directory, "pc_model.jpc"), self.pc_model)

        # CoDD config
        with open(os.path.join(save_directory, "codd_config.json"), "w") as f:
            json.dump(asdict(self.codd_config), f, indent=2)

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload CoddLlada model",
        private: bool = False,
        token: Optional[str] = None,
    ):
        """Save and upload the unified model to HuggingFace Hub."""
        import tempfile
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id, exist_ok=True, private=private, token=token)

        with tempfile.TemporaryDirectory() as tmp_dir:
            self.save_pretrained(tmp_dir)
            api.upload_folder(
                folder_path=tmp_dir,
                repo_id=repo_id,
                commit_message=commit_message,
                token=token,
            )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def decode(
        self,
        output_ids: torch.Tensor,
        prompt_length: int = 0,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode generated token ids to text, optionally stripping the prompt."""
        ids = output_ids.squeeze(0) if output_ids.dim() > 1 else output_ids
        return self.tokenizer.decode(
            ids[prompt_length:], skip_special_tokens=skip_special_tokens
        )

    def _format_prompt(
        self, user_message: str, system_instruction: Optional[str] = None
    ) -> torch.Tensor:
        """Format *user_message* using the LLaDA chat template and tokenize."""
        sys_block = ""
        if system_instruction:
            sys_block = (
                f"<|start_header_id|>system<|end_header_id|>\n\n"
                f"{system_instruction}<|eot_id|>"
            )
        prompt_text = (
            f"<|startoftext|>{sys_block}"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"Problem:\n{user_message}\n\nSolution:<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        return self.tokenizer.encode(prompt_text, return_tensors="pt")
