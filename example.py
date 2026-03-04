import torch
from eval.llada.codd_llada import CoddLlada

torch.cuda.manual_seed(42)
device = torch.device("cuda:0")

# ── Load the unified CoddLlada model (base dLLM + PC) in one call ──
codd = CoddLlada.from_pretrained(
    "GSAI-ML/LLaDA-8B-Instruct",
    pc_model_id="il18/llada-math-pc",
    device_map=device,
)

system_instruction = (
    "You are a helpful assistant. Justify your final answer by first explaining "
    "your step-by-step derivation or reasoning. Conclude by presenting the final "
    "answer in the format: boxed{ANSWER}."
)
question = "For what value of $x$ is $2^3\\cdot3^x=72$?"

prompt = f"""
<|startoftext|><|start_header_id|>system<|end_header_id|>

{system_instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>

Problem:
{question}

Solution:<|eot_id|><|start_header_id|>assistant<|end_header_id|>


"""

prompt_ids = codd.tokenizer.encode(prompt, return_tensors='pt').to(device)

# ── Demo: CoDD native forward pass ──
# codd(input_ids) returns CoDDOutput with .logits
# Without PC (base model only):
base_output = codd(prompt_ids, use_pc=False)
print(f"Base logits shape: {base_output.logits.shape}")

# With PC logit modification on a block range:
pc_output = codd(prompt_ids, pc_block_range=(0, 32), use_pc=True)
print(f"PC-modified logits shape: {pc_output.logits.shape}")

# ── Full generation using codd.generate() ──
# Under the hood this calls a single generation loop that invokes
# codd.forward() at each step — PC modification is transparent.

# Base model generation (no PC guidance):
base_gen = codd.generate(
    prompt_ids,
    use_pc=False,
    num_steps=64,
    gen_length=256,
    block_length=32,
)

# CoDD generation (with PC guidance):
codd_gen = codd.generate(
    prompt_ids,
    use_pc=True,
    num_steps=64,
    gen_length=256,
    block_length=32,
    pc_temperature=0.1,
    pc_frac=0.7,
)

print("=" * 60)
print("PROBLEM: ")
print("=" * 60)
print(question)
print("=" * 60)
print("🔹 BASE MODEL OUTPUT (LLaDA-8B-Instruct)")
print("=" * 60)
print(codd.decode(base_gen, prompt_length=prompt_ids.shape[1]))
print()
print("=" * 60)
print("🔸 CODD OUTPUT (with PC guidance)")
print("=" * 60)
print(codd.decode(codd_gen, prompt_length=prompt_ids.shape[1]))
print()
print("=" * 60)

