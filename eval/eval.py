import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
import argparse
import json
import logging
import torch
from lm_eval import evaluator
from lm_eval.tasks import TaskManager
from harness import DreamEvalHarness, ProfileEvalHarness, LladaEvalHarness
from utils import parse_results
from transformers import AutoModel, AutoTokenizer
import pyjuice as juice
from dream.modeling_dream import DreamModel
from peft import PeftModel
from pc_model_hf import PyJuiceHubModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the tasks you want to support
TASKS = {'gsm8k': 'gsm8k', 'gpqa': 'gpqa_main_generative_n_shot', 'mbpp': 'mbpp_instruct', 'math500': 'minerva_math500'} 

def get_model(args):
    
    model_alias = args.model_alias
    alg = args.alg
    # Canonical names only
    tokens_per_step = args.tokens_per_step
    num_steps = args.num_steps
    task_name = args.task

    logger.info(f"Configuring model details for alias: {model_alias}")
    
    pc_model = None
    if args.pc_ckpt:
        logger.info(f"Loading PC model from {args.pc_ckpt}")
        wrapper_model = PyJuiceHubModel.from_pretrained(args.pc_ckpt)
        pc = wrapper_model.pc
        pc_model = juice.compile(pc)
        pc_model.to(torch.device(f"cuda:0"))
        
    if model_alias == "dream":
        if args.dream_ckpt is None:
            raise ValueError("--dream_ckpt is required when --model_alias=dream")
        dream = DreamModel.from_pretrained(args.dream_ckpt, 
                                               trust_remote_code=True,  
                                               attn_implementation="sdpa", 
                                               torch_dtype=torch.bfloat16, 
                                               device_map="cuda",
                                               local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(args.dream_ckpt, trust_remote_code=True)
        model = DreamEvalHarness(
            pretrained=dream,
            tokenizer=tokenizer,
            alg=alg,
            block_diff=args.dream_block,
            window=args.dream_window,
            num_steps=num_steps,
            max_gen_toks=512 if task_name=="math500" else 256,
            pc_model=pc_model,
            pc_temperature=args.pc_temperature,
            pc_frac=args.pc_frac,
            reverse_frac=args.reverse_frac
        )
        
    elif model_alias == "llada":
                # Load PEFT/LoRA checkpoint if specified

        if args.llada_ckpt is None:
            raise ValueError("--llada_ckpt is required when --model_alias=llada")
        llada = AutoModel.from_pretrained(args.llada_ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16, local_files_only=True).to("cuda").eval()
        tokenizer = AutoTokenizer.from_pretrained(args.llada_ckpt, trust_remote_code=True,local_files_only=True)
        if args.peft_ckpt:
            logger.info(f"Loading PEFT checkpoint from {args.peft_ckpt}")
            llada = PeftModel.from_pretrained(llada, args.peft_ckpt, torch_dtype=torch.bfloat16).to("cuda").eval()
            logger.info("PEFT checkpoint loaded successfully")
        model = LladaEvalHarness(
            pretrained=llada, 
            tokenizer=tokenizer, 
            alg=alg, 
            tokens_per_step=tokens_per_step, 
            num_steps=num_steps,
            pc_model=pc_model,
            pc_temperature=args.pc_temperature,
            pc_frac=args.pc_frac,
            reverse_frac=args.reverse_frac,
            block_length=args.block_length
        )

    else:
        raise ValueError(f"Unknown model alias: {model_alias}.")

    return model

def main():
    parser = argparse.ArgumentParser(description="Evaluate language models using EleutherAI Eval Harness.")
    parser.add_argument(
        "--model_alias",
        required=True,
        choices=["dream", "llada"],
        help="Alias of the model to evaluate."
    )
    
    parser.add_argument(
        "--task",
        required=True,
        choices=["gsm8k", "mbpp", "gpqa",  "math500",],
        help="Please choose one of the following tasks: gsm8k, math, gpqa, humaneval"
    )
    parser.add_argument(
        "--output_dir", 
        default="results",
        help="Directory to save evaluation results json file."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of samples per task for quick testing (e.g., 10 or 0.1 for 10%%)."
    )
    parser.add_argument(
        "--alg",
        choices=["low_confidence", "random", "origin", "entropy", "margin", "margin", "topprob"]
    )
    parser.add_argument("--dream_block", default = False, action = "store_true")
    parser.add_argument("--dream_window", default = False, action = "store_true")

    parser.add_argument("--dream_ckpt", type=str, default="Dream-org/Dream-v0-Instruct-7B", help="Checkpoint or repo id for Dream model")
    parser.add_argument("--llada_ckpt", type=str, default="GSAI-ML/LLaDA-8B-Instruct", help="Checkpoint or repo id for LLaDA model")
    parser.add_argument("--peft_ckpt", type=str, default=None, help="Path to PEFT/LoRA checkpoint to load on top of LLaDA model")
    parser.add_argument("--tokens_per_step", type=int, default=None, help="The number of tokens to generate per step K")

    parser.add_argument("--num_shot", type=int, default=0, help="")
    parser.add_argument("--sample_sid", type=int, default=-1, help="")
    parser.add_argument("--sample_eid", type=int, default=-1, help="")
    parser.add_argument("--block_length", type=int, default=32, help="")
    
    parser.add_argument("--pc_ckpt", type=str, default=None, help="Path to the Probabilistic Circuit checkpoint")
    parser.add_argument("--pc_temperature", type=float, default=0.7, help="Temperature for PC refinement")
    parser.add_argument("--pc_frac", type=float, default=0.3, help="Threshold ratio to trigger PC logic")
    parser.add_argument("--reverse_frac", action="store_true", help="If set, PC triggers when mask_ratio > pc_frac")
    
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None
    )
    
    parser.add_argument(
        "--tag",
        type=str,
        default="",
    )

    args = parser.parse_args()
    
    # Validate algorithm choices based on model alias
    valid_algs = {
        "llada": ["low_confidence", "random", "margin", "entropy", "topprob"],
        "dream": ["entropy", "origin", "margin", "topprob"]
    }
    
    if args.model_alias in valid_algs:
        if args.alg not in valid_algs[args.model_alias]:
            valid_alg_str = ", ".join([str(alg) for alg in valid_algs[args.model_alias]])
            raise ValueError(f"Invalid algorithm '{args.alg}' for model '{args.model_alias}'. "
                           f"Valid algorithms for {args.model_alias}: {valid_alg_str}")
    
    task_manager = TaskManager()
    model = get_model(args)
    os.makedirs(args.output_dir, exist_ok=True)
    
    task_str = args.task
    if args.peft_ckpt:
        task_str += f"_peft_"
    if args.dream_block:
        task_str += "_block"
    if args.dream_window:
        task_str += "_window"
    if args.pc_ckpt: 
        task_str += f"_withPC_{os.path.splitext(os.path.basename(args.pc_ckpt))[0]}"
        if args.pc_frac: task_str += f"_pc_frac_{args.pc_frac}"
        if args.pc_temperature: task_str += f"_pc_temp_{args.pc_temperature}"

    if args.num_steps is not None:
        task_str += f"_num_steps={args.num_steps}"
    if args.tag:
        task_str += f"_{args.tag}"
    output_filename = f"{args.num_shot}shot_{args.model_alias}_{task_str}_s{args.sample_sid}_e{args.sample_eid}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    logger.info(f"Results will be saved to: {output_path}")
    
    
    task_name = args.task
    
    if task_name == "math500":
        system_instruction = "You are a helpful assistant. Justify your final answer by first explaining your step-by-step derivation or reasoning. Conclude by presenting the final answer in the format: boxed{ANSWER}."
    elif task_name == "gpqa":
        system_instruction = "You are a helpful assistant. Justify your final answer by first explaining your step-by-step derivation or reasoning. Conclude by presenting the final answer in the format: (LETTER)."
    else:
        system_instruction = "You are a helpful assistant."

    task_list = [TASKS[task_name]]
    
    results = evaluator.simple_evaluate(
        model=model,
        tasks=task_list,
        task_manager=task_manager,
        batch_size=1,
        limit=args.limit,
        log_samples=True,    
        write_out=True,    
        num_fewshot=args.num_shot, 
        apply_chat_template=True,
        system_instruction=system_instruction,
        gen_kwargs=None,
        confirm_run_unsafe_code=True,
        sample_sid=args.sample_sid if args.sample_sid >= 0 else None,
        sample_eid=args.sample_eid if args.sample_eid >= 0 else None
    )

    results["profile"] = model.get_profile()
    parsed_results = parse_results(results, task_name=TASKS[task_name])
    
    with open(output_path, 'w') as f:
        json.dump(parsed_results, f, indent=4)
    

if __name__ == "__main__":
    main()
    
    
