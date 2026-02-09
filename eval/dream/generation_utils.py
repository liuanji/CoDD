# coding=utf-8
# Copyright 2024 The Dream team, HKUNLP Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import (
    GenerationConfig
)
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    logging,
)
from typing import List
logger = logging.get_logger(__name__)

import numpy as np
import time
import pyjuice as juice

def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0



def add_gumbel_noise(logits, temperature, gumbel_noise=None):

    if temperature == 0:
        return logits, None
    
    if gumbel_noise is None:
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        gumbel_noise = (- torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise, gumbel_noise
    else:
        return logits.exp() / gumbel_noise, gumbel_noise

def sample_with_gumbel(logits, gumbel_noise=None, temperature=0.0, top_p=None, top_k=None):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    

    logits_with_noise, gumbel_noise = add_gumbel_noise(logits, temperature, gumbel_noise)
    x0 = torch.argmax(logits_with_noise, dim=-1)
        
    return gumbel_noise, x0

@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None
    
@dataclass
class ProfileOutput(ModelOutput):
    num_forward_evals: int = None
    num_tokens_generated: int = None
    verification_time: float = None
    total_time: float = None
    acceptance_counts: List[int] = None
    
    
@dataclass
class DreamModelOutputWithProfile(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None
    profile: Optional[ProfileOutput] = None


class DreamGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        # diffusion specific params
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", 'origin')
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)
        # Diffusion parameters (canonical names)
        self.tokens_per_step: Optional[int] = kwargs.pop("tokens_per_step", None)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", False)
        self.output_history: bool = kwargs.pop("output_history", False)

        # Special tokens that can be used at generation time
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def validate(self, is_init=False):
        pass

class DreamGenerationMixin:
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        # Do not call torch.repeat_interleave if expand_size is 1 because it clones
        # the input tensor and thus requires more memory although no change is applied
        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        """Performs validation related to the resulting generated length"""

        # Can't throw warnings/exceptions during compilation
        if is_torchdynamo_compiling():
            return

        # 1. Max length warnings related to poor parameterization
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        input_ids_length,
    ):
        """Prepared max and min length in generation configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        elif has_default_max_length:
            if generation_config.max_length == DreamGenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_length
                max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                if max_position_embeddings is not None:
                    generation_config.max_length = min(generation_config.max_length, max_position_embeddings)

        return generation_config

    def _prepare_generation_config(
        self, generation_config: Optional[DreamGenerationConfig], **kwargs: Dict
    ) -> DreamGenerationConfig:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs. This
        function handles retrocompatibility with respect to configuration files.
        """
        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        using_model_generation_config = False
        if generation_config is None:
            generation_config = DreamGenerationConfig.from_model_config(self.config)
            using_model_generation_config = True

        # `torch.compile` can't compile `copy.deepcopy`, arguments in `kwargs` that are part of `generation_config`
        # will mutate the object with `.update`. As such, passing these arguments through `kwargs` is disabled -- an
        # exception will be raised in `_validate_model_kwargs`
        if not is_torchdynamo_compiling():
            generation_config = copy.deepcopy(generation_config)
            _kwargs = generation_config.update(**kwargs)
            # If `generation_config` is provided, let's fallback ALL special tokens to the default values for the model
            if not using_model_generation_config:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.generation_config.pad_token_id
                if generation_config.mask_token_id is None:
                    generation_config.mask_token_id = self.generation_config.mask_token_id

        return generation_config

    def _prepare_special_tokens(
        self,
        generation_config: DreamGenerationConfig,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Prepares the special tokens for generation, overwriting the generation config with their processed versions
        converted to tensor.

        Note that `generation_config` is changed in place and stops being serializable after this method is called.
        That is no problem if called within `generate` (`generation_config` is a local copy that doesn't leave the
        function). However, if called outside `generate`, consider creating a copy of `generation_config` first.
        """

        # Convert special tokens to tensors
        def _tensor_or_none(token, device=None):
            if token is None:
                return token

            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        mask_token_tensor = _tensor_or_none(generation_config.mask_token_id, device=device)

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        # Set pad token if unset (and there are conditions to do so)
        if pad_token_tensor is None and eos_token_tensor is not None:
            pad_token_tensor = eos_token_tensor[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation.")

        # Update generation config with the updated special tokens tensors
        # NOTE: this must be written into a different attribute name than the one holding the original special tokens
        # (in their non-tensor form), in order to enable end-to-end compilation. See
        # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html#limitations
        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._mask_token_tensor = mask_token_tensor

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[DreamGenerationConfig] = None,
        pc_model = None,
        block_diff = False,
        window = False,
        pc_temperature=None,
        pc_frac=None,
        reverse_frac=None,
        **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        
        verifier_model = kwargs.pop("verifier_model", None)
        
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        generation_config = self._prepare_generation_config(generation_config, **kwargs)
        generation_tokens_hook_func = kwargs.pop("generation_tokens_hook_func", lambda step, x, logits: x)
        generation_logits_hook_func = kwargs.pop("generation_logits_hook_func", lambda step, x, logits: logits)

        # 2. Define model inputs
        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)

        # 3. Prepare `max_length`.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
        
        # 4. Check input_ids
        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )
        if (
            hasattr(generation_config, "pad_token_id") and
            torch.any(input_ids == generation_config.pad_token_id) and 
            attention_mask is None
        ):
            warnings.warn(
                "Padding was detected but no attention mask is passed here. For correct "
                "generation results, please set `attention_mask` when batch-padding inputs.",
                UserWarning,
            )

        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask 
        )
        if block_diff:
            result = self._sample_block_diffusion(
                input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                remask_strategy=generation_config.alg,
                pc_model=pc_model,
                pc_temperature=pc_temperature,
                pc_frac=pc_frac,
                reverse_frac=reverse_frac
            )
        elif window:
            result = self._sample_dynamic_window(
                    input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    remask_strategy=generation_config.alg,
                    pc_model=pc_model,
                    pc_temperature=pc_temperature,
                    pc_frac=pc_frac,
                    reverse_frac=reverse_frac
            )
        else:
            result = self._sample(
                input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                generation_tokens_hook_func=generation_tokens_hook_func,
                generation_logits_hook_func=generation_logits_hook_func
            )
        
        return result

    def _sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        max_length = 512 + input_ids.shape[1]
        
        histories = [] if (return_dict_in_generate and output_history) else None

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"
        
        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

        # this allows user-defined token control of the intermediate steps
        x = generation_tokens_hook_func(None, x, None)
        for i in range(steps):
            mask_index = (x == mask_token_id)
            logits = self(x, attention_mask, tok_idx).logits
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

            # this allows user-defined logits control of the intermediate steps
            logits = generation_logits_hook_func(i, x, logits)

            mask_logits = logits[mask_index]
            t = timesteps[i]
            s = timesteps[i + 1]
        
            if alg == 'origin':
                p_transfer = 1 - s / t if i < steps - 1 else 1
                x0 = torch.zeros_like(x[mask_index], device=self.device, dtype=torch.long) + mask_token_id
                transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
                _, x0[transfer_index_t_s]= sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k)
                x[mask_index] = x0.clone()
            else:
                if alg == 'maskgit_plus':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                elif alg == 'margin':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True)
                elif alg == 'entropy':
                    confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
                else:
                    raise RuntimeError(f"Unknown alg: {alg}")
                num_mask_token = mask_index.sum()
                number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else num_mask_token
                if number_transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_index = torch.topk(confidence, number_transfer_tokens)
                    else:
                        confidence = confidence / alg_temp
                        confidence = F.softmax(confidence, dim=-1)
                        transfer_index = torch.multinomial(confidence, num_samples=number_transfer_tokens)
                    x0_ = torch.zeros_like(x0, device=self.device, dtype=torch.long) + mask_token_id
                    x0_[transfer_index] = x0[transfer_index].clone()
                    x[mask_index] = x0_

            # this allows user-defined token control of the intermediate steps
            x = generation_tokens_hook_func(i, x, logits)
            if histories is not None:
                histories.append(x.clone())
        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x[0,:].unsqueeze(0),
                history=histories,
            )
        else:
            return x[0,:].unsqueeze(0)
    
    def _sample_block_diffusion(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor], # Initial mask for the prompt
        generation_config: DreamGenerationConfig,
        block_length=32, remask_strategy='entropy',
        pc_model=None, pc_temperature=0,
        pc_frac=0, reverse_frac=False, vocab_size=152064
    ) -> torch.LongTensor:
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        mask_token_id = generation_config.mask_token_id

        max_length = 512 + input_ids.shape[1]
        num_blocks = (max_length - input_ids.shape[1]) // block_length
        steps_per_block= steps // num_blocks

        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        histories = [] if (return_dict_in_generate and output_history) else None

        for num_block in range(num_blocks):
            start_idx = input_ids.shape[1] + num_block * block_length
            end_idx = input_ids.shape[1] + (num_block + 1) * block_length

            block_mask_index = (x[:, input_ids.shape[1] + num_block * block_length: input_ids.shape[1] + (num_block + 1) * block_length:] == mask_token_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
            for i in range(steps_per_block):
                mask_index = (x == mask_token_id)
                logits = self(x, "full", None).logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                if pc_model is not None:
                    curr_x = x[:, start_idx:end_idx]
                    mask_ratio = (curr_x == mask_token_id).float().mean()

                    if (not reverse_frac and mask_ratio < pc_frac) or (reverse_frac and mask_ratio > pc_frac):
                        curr_logits = logits[:, start_idx:end_idx, :]
                        
                        if pc_temperature < 1:
                            curr_logits = curr_logits / pc_temperature

                        pc_block_x = curr_x.contiguous().view(-1, block_length) 
                        pc_logits = curr_logits.contiguous().view(-1, block_length, curr_logits.size(-1)).float()
                        external_soft_evi = F.log_softmax(pc_logits, dim=-1)
                        
                        pc_value_mask = ~(pc_block_x == mask_token_id)

                        _ = pc_model(
                            pc_block_x, 
                            external_categorical_logps=external_soft_evi, 
                            extern_product_categorical_mode="unnormalized_ll", 
                            external_categorical_value_mask=pc_value_mask
                        )

                        node_samples = juice.queries.sample(
                            pc_model, 
                            conditional=True, 
                            external_categorical_logps=external_soft_evi,
                            _sample_input_ns=False
                        )

                        layer = pc_model.input_layer_group[0]
                        layer_sid, _ = layer._output_ind_range

                        sampled_logits = external_soft_evi.clone()
                        for j in range(node_samples.size(1)):
                            mask = (node_samples[:, j] >= 0)
                            node_ids = node_samples[mask, j] - layer_sid
                            vids = layer.vids[node_ids, 0]
                            psids = layer.s_pids[node_ids]

                            for k in range(vids.size(0)):
                                sampled_logits[j, vids[k], :] += layer.params[psids[k]:psids[k] + vocab_size].log()
                        
                        logits[:, start_idx:end_idx, :] = sampled_logits
                if alg == 'origin':
                    _, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
                    confidence = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    if alg == 'maskgit_plus':
                        confidence, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
                    elif alg == 'margin':
                        confidence, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True)
                    elif alg == 'entropy':
                        confidence, x0 = sample_tokens(logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
                    else:
                        raise RuntimeError(f"Unknown alg: {alg}")
                    
                confidence[:, end_idx:] = -np.inf # everything before start_idx would already been unmasked
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, confidence, -np.inf)

                for j in range(confidence.shape[0]):
                    num_tokens = num_transfer_tokens[j, i].item()
                    if num_tokens > 0:
                        _, select_index = torch.topk(confidence[j], k=num_tokens)
                        x[j, select_index] = x0[j, select_index]
                if histories is not None:
                    histories.append(x.clone())

        # from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Instruct-7B", trust_remote_code=True)
        # x = F.pad(x, (1, 0), value=tokenizer.bos_token_id)

        # decoded = tokenizer.batch_decode(x)
        # # print(f"DEBUG: Decoded samples: {decoded}")
        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
            )
        else:
            return x
 
    def _sample_dynamic_window(
            self,
            input_ids: torch.LongTensor,
            attention_mask: Optional[torch.LongTensor], # Initial mask for the prompt
            generation_config: DreamGenerationConfig,
            block_length=32, remask_strategy='entropy',
            pc_model=None, pc_temperature=0,
            pc_frac=0, reverse_frac=False, vocab_size=152064
        ) -> torch.LongTensor:
        # Init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        
        max_length = 512 + input_ids.shape[1]
            
        histories = [] if (return_dict_in_generate and output_history) else None

        # Pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        
        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)
        for i in range(steps):
            mask_index = (x == mask_token_id)
            # 1. Base Model Forward Pass
            logits = self(x, "full", None).logits
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1) # Dream Shift

            # =================================================================
            # COPULA / PC ADJUSTMENT BLOCK
            # =================================================================

            run_pc_adjustment = False
            start_idx, end_idx = 0, 0

            t = timesteps[i]
            s = timesteps[i + 1]

            tokens_per_step = (max_length - input_ids.shape[1]) // steps 

            with torch.no_grad():
                # 1. Calculate max probability for every token in the sequence
                if remask_strategy == 'entropy':
                    log_probs = F.log_softmax(logits, dim=-1)
                    probs = log_probs.exp()
                    token_confidences = (probs * log_probs).sum(dim=-1) # [Batch, Seq]
                elif remask_strategy == 'origin':
                    token_confidences = torch.rand((x.shape[0], x.shape[1]), device=x.device)

                
                # 2. Ignore tokens that are already unmasked (set their conf to -1)
                # Assuming Batch=1 for this implementation as discussed
                token_confidences[~mask_index] = -float('inf')
                
                # 3. Find Top k positions
                # We need at least 2 masked tokens to do this check
                if mask_index.sum() >= 2:
                    tokens_per_step = min(mask_index.sum(), tokens_per_step)
                    topk_vals, topk_indices = torch.topk(token_confidences[0], k=tokens_per_step)
                    min_idx = topk_indices.min().item()
                    max_idx = topk_indices.max().item()
                    span = max_idx - min_idx + 1

                    # 4. Check "Close Enough" heuristic
                    if span < block_length:

                        run_pc_adjustment = True

                        # Try putting idx1 at the middle 
                        end_idx = max_idx + block_length // 2
                        start_idx = max_idx - block_length // 2
                        
                        # fall back
                        if end_idx >= x.shape[1]:
                            end_idx = x.shape[1] - 1
                            start_idx = end_idx - block_length
                        if start_idx < 0:
                            start_idx = 0
                            end_idx = start_idx + block_length
                        if start_idx < 0 or end_idx >= x.shape[1]:
                            start_idx = 0
                            end_idx = block_length

            # C. PC Logic
            # Only run if the window is valid and roughly matches the block requirements
            if run_pc_adjustment and (end_idx - start_idx) == block_length:
                curr_x = x[:, start_idx:end_idx]
                mask_ratio = (curr_x == mask_token_id).float().mean()

                # Trigger condition
                if (not reverse_frac and mask_ratio < pc_frac) or (reverse_frac and mask_ratio > pc_frac):
                    curr_logits = logits[:, start_idx:end_idx, :]
                    
                    if pc_temperature > 0 and pc_temperature < 1:
                        curr_logits = curr_logits / pc_temperature

                    # Reshape for PC (Batch * Block, Vocab)
                    pc_block_x = curr_x.contiguous().view(-1, block_length) 
                    pc_logits = curr_logits.contiguous().view(-1, block_length, curr_logits.size(-1)).float()
                    external_soft_evi = F.log_softmax(pc_logits, dim=-1)
                    
                    pc_value_mask = ~(pc_block_x == mask_token_id)

                    # Run PC Forward
                    _ = pc_model(
                        pc_block_x, 
                        external_categorical_logps=external_soft_evi, 
                        extern_product_categorical_mode="unnormalized_ll", 
                        external_categorical_value_mask=pc_value_mask
                    )

                    # Sample/Reason from PC
                    node_samples = juice.queries.sample(
                        pc_model, 
                        conditional=True, 
                        external_categorical_logps=external_soft_evi,
                        _sample_input_ns=False
                    )

                    layer = pc_model.input_layer_group[0]
                    layer_sid, _ = layer._output_ind_range

                    # Inject PC evidence back into logits
                    sampled_logits = external_soft_evi.clone()
                    for j in range(node_samples.size(1)):
                        mask = (node_samples[:, j] >= 0)
                        node_ids = node_samples[mask, j] - layer_sid
                        vids = layer.vids[node_ids, 0]
                        psids = layer.s_pids[node_ids]

                        for k in range(vids.size(0)):
                            # Add PC log-params to the base logits (Product of Experts)
                            sampled_logits[j, vids[k], :] += layer.params[psids[k]:psids[k] + vocab_size].log()
                    
                    # Reshape back and update the main logits tensor
                    logits[:, start_idx:end_idx, :] = sampled_logits.view(node_samples.size(1), block_length, -1)

            # Use boolean indexing to flatten: [Num_Masks, Vocab]
            if alg == 'origin':
                _, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
                confidence = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            elif alg == 'maskgit_plus':
                confidence, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
            elif alg == 'margin':
                confidence, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True)
            elif alg == 'entropy':
                confidence, x0 = sample_tokens(logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
            else:
                raise RuntimeError(f"Unknown alg: {alg}")
            
            if run_pc_adjustment:
                confidence[:, :start_idx] = -1e9
                confidence[:, end_idx:] = -1e9
            confidence[~mask_index] = -1e9 # Set already unmasked tokens to very low confidence
            x0 = torch.where(mask_index, x0, x)

            num_mask_token = mask_index.sum()
            number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else num_mask_token
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], number_transfer_tokens)
                x[j, select_index] = x0[j, select_index]

            if histories is not None:
                histories.append(x.clone())

        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x[0,:].unsqueeze(0),
                history=histories,
            )
        else:
            return x[0,:].unsqueeze(0)
