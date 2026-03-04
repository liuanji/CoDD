import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyjuice as juice
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from transformers import AutoModel, AutoTokenizer
from dataclasses import dataclass, fields, asdict
from typing import Optional, Union, Tuple


class PyJuiceHubModel(PyTorchModelHubMixin):
    """Standalone PyJuice model wrapper for HuggingFace Hub (kept for backward compatibility)."""

    def __init__(self, config=None):
        self.config = config
        self.pc = None

    @classmethod
    def _from_pretrained(cls, model_id, revision, cache_dir, force_download, proxies, resume_download, local_files_only, token, **model_kwargs):
        model_path = hf_hub_download(repo_id=model_id, filename="model.jpc")
        
        pc = juice.load(model_path)
        
        instance = cls()
        instance.pc = pc
        return instance


# ======================================================================
# Standalone PC logit-modification function
# ======================================================================

def apply_pc_logits(
    pc_model,
    input_ids: torch.LongTensor,
    logits: torch.Tensor,
    start_idx: int,
    end_idx: int,
    mask_id: int,
    vocab_size: int,
    pc_temperature: float = 0.1,
    pc_frac: float = 0.7,
    reverse_frac: bool = False,
    record_cudagraph: bool = False,
) -> torch.Tensor:
    """Apply PC logit modification on a block range.

    This is the **single, canonical** implementation of the PC "Product-of-Experts"
    logit adjustment used by both :class:`CoDD` (LLaDA) and Dream
    generation utilities.

    Args:
        pc_model: Compiled PyJuice PC model.
        input_ids: Full token-id tensor of shape ``(B, L)`` (used to read
            the block tokens and compute the mask ratio).
        logits: Logit tensor of shape ``(B, L, V)`` — **modified in-place**.
        start_idx / end_idx: Token-position window to modify.
        mask_id: ``[MASK]`` token id.
        vocab_size: Vocabulary size of the base model.
        pc_temperature: Temperature applied to block logits before PC
            conditioning (values < 1 sharpen the distribution).
        pc_frac: Mask-ratio threshold that controls when PC guidance
            activates.
        reverse_frac: If ``True``, apply PC when ``mask_ratio > pc_frac``
            instead of ``< pc_frac``.
        record_cudagraph: If ``True``, the first PC forward will record a
            CUDA graph for subsequent fast replay.

    Returns:
        The (possibly modified) *logits* tensor.
    """
    block_length = end_idx - start_idx
    curr_x = input_ids[:, start_idx:end_idx]
    mask_ratio = (curr_x == mask_id).float().mean()

    # Check whether the mask-ratio condition triggers PC guidance
    should_apply = (
        (not reverse_frac and mask_ratio < pc_frac)
        or (reverse_frac and mask_ratio > pc_frac)
    )
    if not should_apply:
        return logits

    curr_logits = logits[:, start_idx:end_idx, :]

    if pc_temperature > 0 and pc_temperature < 1:
        curr_logits = curr_logits / pc_temperature

    pc_block_x = curr_x.contiguous().view(-1, block_length)
    pc_logits = curr_logits.contiguous().view(-1, block_length, curr_logits.size(-1)).float()
    external_soft_evi = F.log_softmax(pc_logits, dim=-1)

    pc_value_mask = ~(pc_block_x == mask_id)

    # First call can optionally record a CUDA graph for fast replay
    if record_cudagraph:
        pc_model(
            pc_block_x,
            external_categorical_logps=external_soft_evi,
            extern_product_categorical_mode="unnormalized_ll",
            external_categorical_value_mask=pc_value_mask,
            record_cudagraph=True,
        )

    # Regular forward
    pc_model(
        pc_block_x,
        external_categorical_logps=external_soft_evi,
        extern_product_categorical_mode="unnormalized_ll",
        external_categorical_value_mask=pc_value_mask,
    )

    # Sample from the PC conditionally
    node_samples = juice.queries.sample(
        pc_model,
        conditional=True,
        external_categorical_logps=external_soft_evi,
        _sample_input_ns=False,
    )

    layer = pc_model.input_layer_group[0]
    layer_sid, _ = layer._output_ind_range

    sampled_logits = external_soft_evi.clone()
    for j in range(node_samples.size(1)):
        mask = node_samples[:, j] >= 0
        node_ids = node_samples[mask, j] - layer_sid
        vids = layer.vids[node_ids, 0]
        psids = layer.s_pids[node_ids]

        for k in range(vids.size(0)):
            sampled_logits[j, vids[k], :] += layer.params[
                psids[k] : psids[k] + vocab_size
            ].log()

    logits[:, start_idx:end_idx, :] = sampled_logits

    return logits


# ======================================================================
# Config & Output
# ======================================================================

@dataclass
class CoDDConfig:
    """Configuration for the combined dLLM + PC (CoDD) model.
    
    Args:
        base_model_id: HuggingFace model ID for the base diffusion LLM.
        pc_model_id: HuggingFace model ID for the PyJuice PC model.
        base_model_type: ``"llada"`` or ``"dream"`` — controls forward-pass
            calling convention and optional logit shift.
        mask_id: Token ID used for [MASK] in the diffusion LLM.
        vocab_size: Vocabulary size of the base model.
        num_steps: Number of diffusion sampling steps.
        gen_length: Maximum number of tokens to generate.
        block_length: Block length for semi-autoregressive generation.
        temperature: Gumbel noise temperature (0 = greedy).
        cfg_scale: Classifier-free guidance scale (0 = disabled).
        remasking: Remasking strategy ("low_confidence", "random", "margin", "entropy").
        pc_temperature: Temperature applied to logits before PC conditioning.
        pc_frac: Mask-ratio threshold that controls when PC guidance activates.
        reverse_frac: If True, apply PC when mask_ratio > pc_frac instead of <.
    """
    base_model_id: str = ""
    pc_model_id: str = ""
    base_model_type: str = "llada"   # "llada" | "dream"
    mask_id: int = 126336
    vocab_size: int = 126464
    # Generation defaults
    num_steps: int = 64
    gen_length: int = 256
    block_length: int = 32
    temperature: float = 0.0
    cfg_scale: float = 0.0
    remasking: str = "low_confidence"
    # PC defaults
    pc_temperature: float = 0.1
    pc_frac: float = 0.7
    reverse_frac: bool = False


class CoDDOutput:
    """Return type for :meth:`CoDD.forward`, mirroring HuggingFace model outputs."""

    def __init__(self, logits: torch.Tensor):
        self.logits = logits


# ======================================================================
# CoDD nn.Module
# ======================================================================

class CoDD(nn.Module):
    """Unified dLLM + PC wrapper with a native ``forward()`` that runs the
    base model and applies PC logit modification.

    Supports both **LLaDA** and **Dream** as the base model — controlled by
    ``config.base_model_type``.

    Usage (LLaDA)::

        codd = CoDD.from_pretrained(
            base_model_id="GSAI-ML/LLaDA-8B-Instruct",
            pc_model_id="il18/llada-math-pc",
        )
        output = codd(input_ids, pc_block_range=(s, e))

    Usage (Dream)::

        from eval.dream.modeling_dream import DreamModel

        codd = CoDD.from_pretrained(
            base_model_id="Dream-org/Dream-v0-Instruct-7B",
            pc_model_id="il18/dream-math-pc",
            base_model_class=DreamModel,
            base_model_type="dream",
            mask_id=151666,
            vocab_size=152064,
        )
        output = codd(input_ids, pc_block_range=(s, e))
    """

    def __init__(
        self,
        base_model: nn.Module,
        tokenizer,
        pc_model=None,
        config: Optional[CoDDConfig] = None,
    ):
        super().__init__()
        self.base_model = base_model
        # tokenizer and config are not nn.Modules — store them via
        # object.__setattr__ to avoid nn.Module registration issues.
        object.__setattr__(self, "tokenizer", tokenizer)
        object.__setattr__(self, "codd_config", config or CoDDConfig())
        self.pc_model = pc_model
        self._pc_compiled = False

    @property
    def config(self) -> CoDDConfig:
        return self.codd_config

    @property
    def device(self) -> torch.device:
        return next(self.base_model.parameters()).device

    # ------------------------------------------------------------------
    # Resolve helper
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_file(path_or_repo: str, filename: str) -> Optional[str]:
        """Find *filename* in a local directory or on HuggingFace Hub.

        Returns the resolved local path, or ``None`` if the file does not
        exist in either location.
        """
        local = os.path.join(path_or_repo, filename)
        if os.path.isfile(local):
            return local
        try:
            return hf_hub_download(repo_id=path_or_repo, filename=filename)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        pc_model_id: Optional[str] = None,
        base_model_class=None,
        device: Union[str, torch.device] = "cuda:0",
        torch_dtype=None,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> "CoDD":
        """Load a CoDD model.

        Supports two modes, detected automatically:

        1. **Unified repo** — a single HuggingFace repo (or local directory)
           that contains the base model weights, ``pc_model.jpc``, and
           ``codd_config.json``.  Created by :meth:`save_pretrained`.

           .. code-block:: python

               codd = CoDD.from_pretrained("user/codd-dream-math",
                                           base_model_class=DreamModel)

        2. **Separate repos** (legacy) — the first argument is the base
           model id and ``pc_model_id`` points at a separate PyJuice Hub
           repo.

           .. code-block:: python

               codd = CoDD.from_pretrained(
                   "Dream-org/Dream-v0-Instruct-7B",
                   pc_model_id="il18/dream-math-pc",
                   base_model_class=DreamModel,
               )

        Args:
            pretrained_model_name_or_path: Local directory or HuggingFace
                repo id.  In unified mode this single path contains
                everything; in legacy mode it is the base-model id.
            pc_model_id: (Legacy mode only) HuggingFace repo id for a
                separate PyJuice PC model.
            base_model_class: Model class for the base dLLM.  Defaults to
                ``AutoModel``.  Pass ``DreamModel`` for Dream.
            device: Target device (ignored when ``device_map`` is given).
            torch_dtype: Torch dtype for model weights.
            trust_remote_code: Forwarded to ``from_pretrained`` calls.

        Accepts any :class:`CoDDConfig` field as a keyword argument
        (``base_model_type``, ``mask_id``, ``vocab_size``, …).  Extra
        kwargs are forwarded to the base model's ``from_pretrained``
        (``attn_implementation``, ``device_map``, ``local_files_only``, …).
        """
        device = torch.device(device) if isinstance(device, str) else device

        # Separate CoDDConfig fields from model-loading kwargs
        codd_field_names = {f.name for f in fields(CoDDConfig)}
        config_overrides = {k: v for k, v in kwargs.items() if k in codd_field_names}
        model_extra = {k: v for k, v in kwargs.items() if k not in codd_field_names}

        # --- Detect unified repo (has codd_config.json) ---
        codd_config_path = cls._resolve_file(
            pretrained_model_name_or_path, "codd_config.json"
        )

        if codd_config_path is not None:
            # ===== Unified repo mode =====
            with open(codd_config_path) as f:
                saved_config = json.load(f)
            saved_config.update(config_overrides)  # caller can override
            cfg = CoDDConfig(**saved_config)

            model_cls = base_model_class or AutoModel
            model_kwargs = {"trust_remote_code": trust_remote_code, **model_extra}
            if torch_dtype is not None:
                model_kwargs["torch_dtype"] = torch_dtype

            base_model = model_cls.from_pretrained(
                pretrained_model_name_or_path, **model_kwargs
            )
            if "device_map" not in model_extra:
                base_model = base_model.to(device)

            tokenizer_kwargs = {"trust_remote_code": trust_remote_code}
            if "local_files_only" in model_extra:
                tokenizer_kwargs["local_files_only"] = model_extra["local_files_only"]
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, **tokenizer_kwargs
            )

            # PC — lives in the same repo as pc_model.jpc
            pc_model = None
            pc_path = cls._resolve_file(
                pretrained_model_name_or_path, "pc_model.jpc"
            )
            if pc_path is not None:
                pc_raw = juice.load(pc_path)
                pc_device = (
                    device if "device_map" not in model_extra
                    else next(base_model.parameters()).device
                )
                pc_model = juice.compile(pc_raw).to(pc_device)

            return cls(base_model, tokenizer, pc_model=pc_model, config=cfg)

        # ===== Legacy separate-repos mode =====
        base_model_id = pretrained_model_name_or_path
        cfg = CoDDConfig(
            base_model_id=base_model_id,
            pc_model_id=pc_model_id or "",
            **config_overrides,
        )

        model_cls = base_model_class or AutoModel
        model_kwargs = {"trust_remote_code": trust_remote_code, **model_extra}
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        base_model = model_cls.from_pretrained(base_model_id, **model_kwargs)
        if "device_map" not in model_extra:
            base_model = base_model.to(device)

        tokenizer_kwargs = {"trust_remote_code": trust_remote_code}
        if "local_files_only" in model_extra:
            tokenizer_kwargs["local_files_only"] = model_extra["local_files_only"]
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, **tokenizer_kwargs)

        pc_model = None
        if pc_model_id is not None:
            pc_raw = PyJuiceHubModel.from_pretrained(pc_model_id).pc
            pc_device = (
                device if "device_map" not in model_extra
                else next(base_model.parameters()).device
            )
            pc_model = juice.compile(pc_raw).to(pc_device)

        return cls(base_model, tokenizer, pc_model=pc_model, config=cfg)

    # ------------------------------------------------------------------
    # Save / Push
    # ------------------------------------------------------------------

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save the unified CoDD model to a single directory.

        The directory will contain:

        * Base model weights and config (standard HuggingFace format)
        * Tokenizer files
        * ``pc_model.jpc`` — the PyJuice probabilistic circuit
        * ``codd_config.json`` — :class:`CoDDConfig` metadata

        The resulting directory can be reloaded with
        ``CoDD.from_pretrained(save_directory)`` and pushed to the Hub
        with ``CoDD.push_to_hub()``.
        """
        os.makedirs(save_directory, exist_ok=True)

        # 1. Base model (weights + model config)
        self.base_model.save_pretrained(save_directory, **kwargs)

        # 2. Tokenizer
        self.tokenizer.save_pretrained(save_directory)

        # 3. PC model
        if self.pc_model is not None:
            pc_path = os.path.join(save_directory, "pc_model.jpc")
            juice.save(pc_path, self.pc_model)

        # 4. CoDD config
        config_path = os.path.join(save_directory, "codd_config.json")
        with open(config_path, "w") as f:
            json.dump(asdict(self.codd_config), f, indent=2)

    def push_to_hub(self, repo_id: str, commit_message: str = "Upload CoDD model", private: bool = False, token: Optional[str] = None):
        """Save and upload the unified CoDD model to HuggingFace Hub.

        Creates the repository if it does not exist, saves all files to a
        temporary directory, and uploads the entire folder.

        Args:
            repo_id: HuggingFace Hub repository id (e.g. ``"user/model-name"``).
            commit_message: Git commit message for the upload.
            private: Whether the repository should be private.
            token: HuggingFace API token (uses cached token if ``None``).
        """
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
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.LongTensor,
        pc_block_range: Optional[Tuple[int, int]] = None,
        use_pc: bool = True,
        # --- LLaDA-specific ---
        cfg_scale: float = 0.0,
        prompt_index: Optional[torch.BoolTensor] = None,
        # --- Dream-specific ---
        attention_mask=None,
        tok_idx=None,
        # --- PC overrides ---
        pc_temperature: Optional[float] = None,
        pc_frac: Optional[float] = None,
        reverse_frac: Optional[bool] = None,
    ) -> CoDDOutput:
        """Run base dLLM forward and optionally apply PC logit modification.

        The calling convention of the *base model* is chosen automatically
        based on ``config.base_model_type``:

        * **llada** — ``base_model(input_ids).logits`` with optional CFG.
        * **dream** — ``base_model(input_ids, attention_mask, tok_idx).logits``
          followed by the Dream logit shift.

        Args:
            input_ids: Token ids ``(B, L)``.
            pc_block_range: ``(start_idx, end_idx)`` window for PC modification.
            use_pc: Master switch for PC guidance.
            cfg_scale: LLaDA classifier-free guidance scale.
            prompt_index: Boolean mask for CFG (LLaDA only).
            attention_mask: Attention mask (Dream only — ``"full"`` or tensor).
            tok_idx: Token index tensor (Dream only).
            pc_temperature / pc_frac / reverse_frac: Override config defaults.

        Returns:
            :class:`CoDDOutput` with ``.logits`` of shape ``(B, L, V)``.
        """
        cfg = self.config
        pc_temperature = pc_temperature if pc_temperature is not None else cfg.pc_temperature
        pc_frac = pc_frac if pc_frac is not None else cfg.pc_frac
        reverse_frac = reverse_frac if reverse_frac is not None else cfg.reverse_frac
        mask_id = cfg.mask_id
        vocab_size = cfg.vocab_size

        # --- 1. Base dLLM forward ---
        if cfg.base_model_type == "dream":
            logits = self._dream_forward(input_ids, attention_mask, tok_idx)
        else:
            logits = self._llada_forward(input_ids, cfg_scale, prompt_index, mask_id)

        # --- 2. PC logit modification ---
        if use_pc and self.pc_model is not None and pc_block_range is not None:
            record = not self._pc_compiled
            logits = apply_pc_logits(
                pc_model=self.pc_model,
                input_ids=input_ids,
                logits=logits,
                start_idx=pc_block_range[0],
                end_idx=pc_block_range[1],
                mask_id=mask_id,
                vocab_size=vocab_size,
                pc_temperature=pc_temperature,
                pc_frac=pc_frac,
                reverse_frac=reverse_frac,
                record_cudagraph=record,
            )
            if record:
                self._pc_compiled = True

        return CoDDOutput(logits=logits)

    # ------------------------------------------------------------------
    # Base-model forward helpers
    # ------------------------------------------------------------------

    def _llada_forward(self, input_ids, cfg_scale, prompt_index, mask_id):
        """LLaDA forward with optional classifier-free guidance."""
        if cfg_scale > 0.0 and prompt_index is not None:
            un_x = input_ids.clone()
            un_x[prompt_index] = mask_id
            x_ = torch.cat([input_ids, un_x], dim=0)
            base_out = self.base_model(x_)
            logits, un_logits = torch.chunk(base_out.logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = self.base_model(input_ids).logits
        return logits

    def _dream_forward(self, input_ids, attention_mask, tok_idx):
        """Dream forward with the mandatory logit shift."""
        if attention_mask is None:
            attention_mask = "full"
        logits = self.base_model(input_ids, attention_mask, tok_idx).logits
        # Dream logit shift: first position keeps its logit, rest are shifted
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        return logits

    # ------------------------------------------------------------------
    # High-level generation (LLaDA path)
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
        """Run diffusion generation (LLaDA path by default).

        For Dream-based generation, use Dream's native
        ``model.diffusion_generate()`` with a :class:`CoDD` instance as the
        model — its ``forward()`` handles the PC modification transparently.

        Args:
            prompt: String or token-id tensor ``(1, L)``.
            generate_fn: Generation function to use. If ``None``, auto-imports
                ``llada_diffusion_generate`` from the eval package.
            use_pc: Activate PC-guided CoDD generation.
            system_instruction: Optional system prompt (string prompts only).
            Remaining kwargs override :class:`CoDDConfig` defaults.

        Returns:
            Token-id tensor ``(1, prompt_length + gen_length)``.
        """
        if generate_fn is None:
            # Auto-import: try both possible paths depending on working directory
            try:
                from eval.llada.llada_generate import llada_diffusion_generate
            except ImportError:
                from llada.llada_generate import llada_diffusion_generate
            generate_fn = llada_diffusion_generate

        cfg = self.config

        num_steps = num_steps if num_steps is not None else cfg.num_steps
        gen_length = gen_length if gen_length is not None else cfg.gen_length
        block_length = block_length if block_length is not None else cfg.block_length
        temperature = temperature if temperature is not None else cfg.temperature
        cfg_scale = cfg_scale if cfg_scale is not None else cfg.cfg_scale
        remasking = remasking if remasking is not None else cfg.remasking

        if isinstance(prompt, str):
            prompt = self._apply_chat_template(prompt, system_instruction)

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
    # Utility
    # ------------------------------------------------------------------

    def decode(self, output_ids: torch.Tensor, prompt_length: int = 0, skip_special_tokens: bool = True) -> str:
        """Decode generated token ids to text, optionally stripping the prompt."""
        ids = output_ids.squeeze(0) if output_ids.dim() > 1 else output_ids
        return self.tokenizer.decode(ids[prompt_length:], skip_special_tokens=skip_special_tokens)

    def _apply_chat_template(self, user_message: str, system_instruction: Optional[str] = None) -> torch.Tensor:
        """Format *user_message* using the LLaDA-style chat template and tokenize."""
        sys_block = ""
        if system_instruction:
            sys_block = (
                f"<|start_header_id|>system<|end_header_id|>\n\n"
                f"{system_instruction}<|eot_id|>"
            )

        prompt_text = (
            f"<|startoftext|>{sys_block}"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"Problem:\n{user_message}\n\n"
            f"Solution:<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        return self.tokenizer.encode(prompt_text, return_tensors="pt")
