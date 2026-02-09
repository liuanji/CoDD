import subprocess
import threading
import queue
import time

# --- CONFIGURATION ---
GPUS = [0]  # List of GPU IDs to use
EVAL_SCRIPT = "./eval.sh"

# --- JOB LIST ---
# I have pre-filled this with the commands you provided.
# Note: I preserved the exact text, including the likely typo "gsm8" in the final batch.
JOBS = [
    # '--model_alias llada --task gsm8k --alg low_confidence --num_steps 256 --num_shot 0  --limit 2',
    # '--model_alias llada --task math500 --alg low_confidence --num_steps 128 --num_shot 0  --limit 2',
    # '--model_alias llada --task gpqa --alg low_confidence --num_steps 64 --num_shot 0 --limit 2',
    # '--model_alias llada --task mbpp --alg low_confidence --num_steps 64 --num_shot 0 --limit 2',

    # '--model_alias llada --task gsm8k --alg low_confidence --num_steps 256 --num_shot 0  --pc_ckpt /scratch2/ianli18/data/test1221.jpc --pc_temperature 0.1 --pc_frac 0.3 --limit 2',
    '--model_alias llada --task math500 --alg low_confidence --num_steps 256 --num_shot 0 --pc_ckpt il18/llada-math-pc --pc_temperature 0.1 --pc_frac 0.3',
    # '--model_alias llada --task gpqa --alg low_confidence --num_steps 64 --num_shot 0 --pc_ckpt /scratch2/ianli18/data/test1221.jpc --pc_temperature 0.1 --pc_frac 0.3 --limit 2',
    # '--model_alias llada --task mbpp --alg low_confidence --num_steps 64 --num_shot 0 --pc_ckpt /scratch2/ianli18/data/test1221.jpc --pc_temperature 0.1 --pc_frac 0.3 --limit 2',
    
    # '--model_alias dream --task gpqa --dream_window --alg entropy --num_steps 256 --num_shot 0  --pc_ckpt /scratch2/ianli18/data/best_dream_math_converted.jpc --pc_temperature 0.2 --pc_frac 0.3 --limit 2',
    # '--model_alias dream --task math500 --dream_window --alg entropy --num_steps 128 --num_shot 0  --pc_ckpt /scratch2/ianli18/data/best_dream_math_converted.jpc --pc_temperature 0.1 --pc_frac 0.3 --limit 2',
    # '--model_alias dream --task gsm8k --dream_window --alg entropy --num_steps 64 --num_shot 0  --pc_ckpt /scratch2/ianli18/data/best_dream_math_converted.jpc --pc_temperature 0.1 --pc_frac 0.3 --limit 2',
    # '--model_alias dream --task mbpp --dream_window --alg entropy --num_steps 64 --num_shot 0  --pc_ckpt /scratch2/ianli18/data/best_dream_math_converted.jpc --pc_temperature 0.1 --pc_frac 0.3 --limit 2',

    # '--model_alias dream --task gpqa --dream_block --alg entropy --num_steps 256 --num_shot 0  --pc_ckpt /scratch2/ianli18/data/best_dream_math_converted.jpc --pc_temperature 0.2 --pc_frac 0.3 --limit 2',
    # '--model_alias dream --task math500 --dream_block --alg entropy --num_steps 128 --num_shot 0  --pc_ckpt /scratch2/ianli18/data/best_dream_math_converted.jpc --pc_temperature 0.1 --pc_frac 0.3 --limit 2',
    # '--model_alias dream --task gsm8k --dream_block --alg entropy --num_steps 64 --num_shot 0  --pc_ckpt /scratch2/ianli18/data/best_dream_math_converted.jpc --pc_temperature 0.1 --pc_frac 0.3 --limit 2',
    # '--model_alias dream --task mbpp --dream_block --alg entropy --num_steps 64 --num_shot 0  --pc_ckpt /scratch2/ianli18/data/best_dream_math_converted.jpc --pc_temperature 0.1 --pc_frac 0.3 --limit 2',

    # '--model_alias dream --task gpqa --alg entropy --num_steps 256 --num_shot 0  --limit 2',
    # '--model_alias dream --task gsm8k --alg entropy --num_steps 128 --num_shot 0  --limit 2',
    # '--model_alias dream --task math500 --alg entropy --num_steps 64 --num_shot 0  --limit 2',
    # '--model_alias dream --task mbpp --alg entropy --num_steps 64 --num_shot 0  --limit 2',
]

# --- WORKER FUNCTION ---
def gpu_worker(gpu_id, job_queue):
    while True:
        try:
            # Get next job from queue, don't wait if empty
            job_args = job_queue.get_nowait()
        except queue.Empty:
            print(f"GPU {gpu_id}: No more jobs. Exiting.")
            break
        
        print(f"GPU {gpu_id}: Starting job...")
        
        # Construct the full command
        # This assumes eval.sh can accept a single GPU like '--gpus 0'
        # We wrap job_args in single quotes just to be safe, though subprocess handles most of it.
        full_command = f"{EVAL_SCRIPT} --gpus {gpu_id} --run '{job_args}'"
        
        try:
            # shell=True is used here to simplify passing the entire argument string
            subprocess.run(full_command, shell=True, check=True)
            print(f"GPU {gpu_id}: Job finished.")
        except subprocess.CalledProcessError as e:
            print(f"GPU {gpu_id}: Job failed with error: {e}")
        
        job_queue.task_done()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    job_queue = queue.Queue()
    
    # Fill the queue
    for job in JOBS:
        job_queue.put(job)
    
    print(f"Loaded {job_queue.qsize()} jobs into queue.")
    
    threads = []
    
    # Create a thread for each GPU
    for gpu in GPUS:
        t = threading.Thread(target=gpu_worker, args=(gpu, job_queue))
        t.start()
        threads.append(t)
    
    # Wait for all threads to finish
    for t in threads:
        t.join()
        
    print("All jobs completed.")