import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import pyjuice as juice


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


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


def _is_codd_model(model):
    """Check whether *model* is a CoDD instance with a PC actually loaded."""
    return (hasattr(model, 'pc_model') and hasattr(model, 'codd_config') and hasattr(model, '_pc_compiled'))


@torch.no_grad()
def llada_diffusion_generate(
    model, prompt, num_steps=128, gen_length=128, block_length=128,
    temperature=0., cfg_scale=0., remasking='low_confidence', mask_id=126336,
    # ---- extra kwargs used when model is a CoDD instance ----
    use_pc=False, pc_temperature=None, pc_frac=None, reverse_frac=None,
):
    '''Unified LLaDA diffusion generation.

    Works with both a plain base model (original behaviour) and with a
    :class:`CoDD` model whose ``forward()`` natively handles PC logit
    modification when ``pc_block_range`` is supplied.

    Args:
        model: Either a plain dLLM or a :class:`CoDD` instance.
        prompt: A tensor of shape ``(1, L)``.
        num_steps: Total sampling steps across all blocks.
        gen_length: Number of tokens to generate.
        block_length: Block length for semi-autoregressive generation.
        temperature: Gumbel noise temperature (0 = greedy argmax).
        cfg_scale: Classifier-free guidance scale (0 = disabled).
        remasking: ``'low_confidence'``, ``'random'``, ``'margin'``, or ``'entropy'``.
        mask_id: Token id for ``[MASK]``.
        use_pc: If ``True`` **and** *model* is a CoDD instance, activate
            PC logit modification inside ``forward()``.
        pc_temperature: Forwarded to ``CoDD.forward()``.
        pc_frac: Forwarded to ``CoDD.forward()``.
        reverse_frac: Forwarded to ``CoDD.forward()``.
    '''
    is_codd = _is_codd_model(model)

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert num_steps % num_blocks == 0
    steps = num_steps // num_blocks

    for num_block in tqdm(range(num_blocks), desc="Processing Blocks"):
        start_idx = prompt.shape[1] + num_block * block_length
        end_idx = prompt.shape[1] + (num_block + 1) * block_length

        block_mask_index = (x[:, start_idx:end_idx] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        for i in range(steps):
            mask_index = (x == mask_id)

            # ---------- forward pass ----------
            if is_codd:
                # CoDD.forward() handles CFG + PC modification internally
                pc_block_range = (start_idx, end_idx) if use_pc else None
                output = model(
                    x,
                    pc_block_range=pc_block_range,
                    use_pc=use_pc,
                    cfg_scale=cfg_scale,
                    prompt_index=prompt_index if cfg_scale > 0. else None,
                    pc_temperature=pc_temperature,
                    pc_frac=pc_frac,
                    reverse_frac=reverse_frac,
                )
                logits = output.logits
            else:
                # Plain base model path (backward compatible)
                if cfg_scale > 0.:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x).logits

            # ---------- sampling ----------
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            elif remasking == 'margin':
                probs = torch.softmax(logits_with_noise, dim=-1)
                top_probs, _ = torch.topk(probs, k=2, dim=-1)
                x0_p = top_probs[..., 0] - top_probs[..., 1]
            elif remasking == 'entropy':
                probs = torch.softmax(logits_with_noise, dim=-1)
                epsilon = 1e-10
                log_probs = torch.log(probs + epsilon)
                x0_p = torch.sum(probs * log_probs, dim=-1)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, end_idx:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            for j in range(confidence.shape[0]):
                num_tokens = num_transfer_tokens[j, i].item()
                if num_tokens > 0:
                    _, select_index = torch.topk(confidence[j], k=num_tokens)
                    x[j, select_index] = x0[j, select_index]

    return x


# ---------------------------------------------------------------------------
# Legacy wrapper kept for backward compatibility
# ---------------------------------------------------------------------------

@torch.no_grad()
def llada_diffusion_pc_generate(model, prompt, num_steps=256, gen_length=512, block_length=32, temperature=0.,
    cfg_scale=0., remasking='low_confidence', mask_id=126336, pc_model=None, pc_temperature=0.7, pc_frac=0.3,
    reverse_frac=False, vocab_size=126464
):
    '''
    LLaDA Diffusion generation with PC (legacy interface).

    If *model* is a :class:`CoDD` instance, this simply delegates to
    :func:`llada_diffusion_generate` with ``use_pc=True``.

    Otherwise it runs the original standalone PC-generation loop for full
    backward compatibility.
    '''

    # --- Fast path: CoDD model handles PC natively in forward() ---
    if _is_codd_model(model):
        return llada_diffusion_generate(
            model, prompt,
            num_steps=num_steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking=remasking,
            mask_id=mask_id,
            use_pc=True,
            pc_temperature=pc_temperature,
            pc_frac=pc_frac,
            reverse_frac=reverse_frac,
        )

    # --- Legacy path: separate base model + pc_model objects ---
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert num_steps % num_blocks == 0
    steps_per_block = num_steps // num_blocks

    compile_flag = False

    for num_block in tqdm(range(num_blocks), desc="Processing Blocks"):
        start_idx = prompt.shape[1] + num_block * block_length
        end_idx = prompt.shape[1] + (num_block + 1) * block_length
        
        block_mask_index = (x[:, start_idx:end_idx] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        
        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            if pc_model is not None:
                curr_x = x[:, start_idx:end_idx]
                mask_ratio = (curr_x == mask_id).float().mean()

                if (not reverse_frac and mask_ratio < pc_frac) or (reverse_frac and mask_ratio > pc_frac):
                    curr_logits = logits[:, start_idx:end_idx, :]
                    
                    if pc_temperature < 1:
                        curr_logits = curr_logits / pc_temperature

                    pc_block_x = curr_x.contiguous().view(-1, block_length)

                    pc_logits = curr_logits.contiguous().view(-1, block_length, curr_logits.size(-1)).float()
                    external_soft_evi = F.log_softmax(pc_logits, dim=-1)
                    
                    pc_value_mask = ~(pc_block_x == mask_id)

                    if not compile_flag:
                        lls = pc_model(pc_block_x, 
                            external_categorical_logps=external_soft_evi, 
                            extern_product_categorical_mode="unnormalized_ll", 
                            external_categorical_value_mask=pc_value_mask, 
                            record_cudagraph = True
                        )
                        compile_flag = True

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

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            elif remasking == 'margin':
                probs = torch.softmax(logits_with_noise, dim=-1)
                top_probs, _ = torch.topk(probs, k=2, dim=-1)
                top1_probs = top_probs[..., 0] 
                top2_probs = top_probs[..., 1]
                x0_p = top1_probs - top2_probs 
            elif remasking == 'entropy':
                probs = torch.softmax(logits_with_noise, dim=-1)
                epsilon = 1e-10
                log_probs = torch.log(probs + epsilon)
                x0_p = torch.sum(probs * log_probs, dim=-1)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, end_idx:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            for j in range(confidence.shape[0]):
                num_tokens = num_transfer_tokens[j, i].item()
                if num_tokens > 0:
                    _, select_index = torch.topk(confidence[j], k=num_tokens)
                    x[j, select_index] = x0[j, select_index]

    return x[0,:].unsqueeze(0)
