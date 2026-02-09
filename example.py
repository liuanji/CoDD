import torch
import pyjuice as juice
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from transformers import AutoModel, AutoTokenizer
from eval.pc_model_hf import PyJuiceHubModel
from eval.llada.llada_generate import llada_diffusion_generate, llada_diffusion_pc_generate

torch.cuda.manual_seed(42)
device = torch.device("cuda:0")

model = AutoModel.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)

pc_model = juice.compile(PyJuiceHubModel.from_pretrained("il18/llada-math-pc").pc).to(device)

system_instruction = "You are a helpful assistant. Justify your final answer by first explaining your step-by-step derivation or reasoning. Conclude by presenting the final answer in the format: boxed{ANSWER}."
question = "For what value of $x$ is $2^3\\cdot3^x=72$?"
prompt = f"""
<|startoftext|><|start_header_id|>system<|end_header_id|>

{system_instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>

Problem:
{question}

Solution:<|eot_id|><|start_header_id|>assistant<|end_header_id|>


"""

prompt_ids = tokenizer.encode(prompt, return_tensors='pt')

base_model_output = llada_diffusion_generate(model, prompt_ids, num_steps=64, gen_length=256, block_length=32, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336)

codd_output = llada_diffusion_pc_generate(model, prompt_ids, num_steps=64, gen_length=256, block_length=32, temperature=0.,
    cfg_scale=0., remasking='low_confidence', mask_id=126336, pc_model=pc_model, pc_temperature=0.1, pc_frac=0.7,
    reverse_frac=False, vocab_size=126464
)

print("=" * 60)
print("PROBLEM: ")
print("=" * 60)
print(question)
print("=" * 60)
print("🔹 BASE MODEL OUTPUT (LLaDA-8B-Instruct)")
print("=" * 60)
print(tokenizer.decode(base_model_output[0, prompt_ids.shape[1]:], skip_special_tokens=True))
print()
print("=" * 60)
print("🔸 CODD OUTPUT (with PC guidance)")
print("=" * 60)
print(tokenizer.decode(codd_output[0, prompt_ids.shape[1]:], skip_special_tokens=True))
print()
print("=" * 60)

