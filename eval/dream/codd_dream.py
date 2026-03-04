"""
CoddDream — DreamModel extended with PyJuice PC guidance.

Inherits all Dream functionality (including ``diffusion_generate``) and adds:
- Automatic PC model storage and passthrough during generation
- Unified ``save_pretrained()`` / ``from_pretrained()`` for base model + PC
- ``push_to_hub()`` for easy sharing
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
from huggingface_hub import hf_hub_download

from .modeling_dream import DreamModel
from .configuration_dream import DreamConfig
from .generation_utils import DreamGenerationConfig

# Ensure the project root is on sys.path so `codd` package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from codd.model_class import CoDDConfig, CoDDOutput, apply_pc_logits, PyJuiceHubModel


class CoddDream(DreamModel):
    """Dream model with integrated PyJuice Probabilistic Circuit (PC).

    This is a *proper subclass* of :class:`DreamModel` — it IS a
    ``PreTrainedModel`` with all the standard HuggingFace API surface.
    It also inherits :class:`DreamGenerationMixin` so
    ``diffusion_generate()`` works out of the box; when a PC is loaded
    it is automatically passed through to the generation loop.

    The PC model is stored as a submodule but excluded from the regular
    state-dict; the PC is persisted separately as ``pc_model.jpc`` via
    PyJuice, and CoDD metadata is saved as ``codd_config.json``.

    Usage::

        # ── From separate repos ──
        model = CoddDream.from_pretrained(
            "Dream-org/Dream-v0-Instruct-7B",
            pc_model_id="il18/dream-math-pc",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

        # ── From unified repo (created by save_pretrained) ──
        model = CoddDream.from_pretrained("user/codd-dream-math")

        # ── Generation (PC applied automatically) ──
        out = model.diffusion_generate(input_ids, ...)

        # ── Save / push ──
        model.save_pretrained("./my-codd-dream")
        model.push_to_hub("user/codd-dream-math")
    """

    config_class = DreamConfig
    _keys_to_ignore_on_load_missing = [r"pc_model\."]
    _keys_to_ignore_on_load_unexpected = [r"pc_model\."]

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def __init__(self, config: DreamConfig):
        super().__init__(config)
        object.__setattr__(
            self, "codd_config",
            CoDDConfig(base_model_type="dream", mask_id=151666, vocab_size=152064),
        )
        object.__setattr__(self, "tokenizer", None)
        self.pc_model = None
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
        """Load a CoddDream model.

        Supports two modes, auto-detected:

        1. **Unified repo** — a single path/repo containing the base
           Dream weights *and* ``pc_model.jpc`` + ``codd_config.json``.
        2. **Separate repos** — ``pretrained_model_name_or_path`` is the
           base Dream model and ``pc_model_id`` points at a separate
           PyJuice Hub repo.

        Any :class:`CoDDConfig` field (``mask_id``, ``vocab_size``, …)
        may be passed as a keyword argument; everything else is forwarded
        to ``DreamModel.from_pretrained``.
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
                base_model_type="dream",
                mask_id=151666,
                vocab_size=152064,
                **config_overrides,
            )

        # --- Load base Dream weights via the standard HF path ---
        # DreamPreTrainedModel.from_pretrained also loads generation_config
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
    # diffusion_generate — auto-pass PC model
    # ------------------------------------------------------------------

    def diffusion_generate(
        self,
        inputs=None,
        generation_config=None,
        use_pc: bool = True,
        **kwargs,
    ):
        """Dream diffusion generation with automatic PC passthrough.

        When ``use_pc=True`` (default) and a PC model is loaded, it is
        automatically forwarded to the generation loop.  All other
        arguments are passed through to
        :meth:`DreamModel.diffusion_generate`.

        PC-specific kwargs (``pc_temperature``, ``pc_frac``,
        ``reverse_frac``) default to the values in :attr:`codd_config`
        but can be overridden per call.
        """
        # Fill in PC defaults from codd_config if not supplied
        pc_model = kwargs.pop("pc_model", None)
        if pc_model is None and use_pc:
            pc_model = self.pc_model

        cfg = self.codd_config
        pc_temperature = kwargs.pop("pc_temperature", cfg.pc_temperature)
        pc_frac = kwargs.pop("pc_frac", cfg.pc_frac)
        reverse_frac = kwargs.pop("reverse_frac", cfg.reverse_frac)

        return super().diffusion_generate(
            inputs=inputs,
            generation_config=generation_config,
            pc_model=pc_model,
            pc_temperature=pc_temperature,
            pc_frac=pc_frac,
            reverse_frac=reverse_frac,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Save / Push
    # ------------------------------------------------------------------

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save the unified CoddDream model.

        Writes:
        - Base model weights + config (via ``DreamModel.save_pretrained``)
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
        commit_message: str = "Upload CoddDream model",
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
