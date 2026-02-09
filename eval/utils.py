import numpy as np
import re
from math_verify import parse, verify

def extract_boxed_answer(text):
    if not text:
        return None
    
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    start_brace = text.find("{", idx)
    if start_brace == -1:
        return None
    count = 0
    for i in range(start_brace, len(text)):
        if text[i] == "{":
            count += 1
        elif text[i] == "}":
            count -= 1
            if count == 0:
                return text[start_brace + 1 : i]
    return None

def parse_results(results, task_name):
    output_data = {
        'accuracies': {},
        'profile': results.get("profile", {}),
        'samples': {}
    }

    if 'results' in results:
        for task, task_results in results['results'].items():
            output_data['accuracies'][task] = {}
            for metric, value in task_results.items():
                if any(m in metric for m in ['exact_match', 'pass@1', 'pass_at_1', 'math_verify']):
                    output_data['accuracies'][task][metric] = value
                    
    if task_name == "math":
        accuracies = []
        for subject, metrics in output_data['accuracies'].items():
            if "hendrycks_math_" in subject:
                val = metrics.get("exact_match,flexible-extract")
                if val is not None: accuracies.append(val)
        if accuracies:
            output_data['accuracies']['hendrycks_math'] = {"aggregate_accuracy": np.mean(accuracies)}

    if 'samples' in results:
        for task, sample_list in results['samples'].items():
            output_data['samples'][task] = []
            
            for sample in sample_list:
                current_filter = sample.get('filter')
                if current_filter not in ['flexible-extract', 'create_test', 'none', 'extract_code', None]:
                    continue

                generation = sample.get('resps', [[""]])[0][0].strip()
                doc = sample.get('doc', {})
                ground_truth = doc.get('answer', sample.get('target', ""))

                if 'math' in task_name:
                    is_correct_val = sample.get('math_verify', -1)
                    extracted = extract_boxed_answer(generation)
                else:
                    metric = sample.get('metrics')[0]
                    is_correct_val = sample.get(metric, None)
                    extracted = sample.get('filtered_resps', [None])[0]

                if task_name == "gsm8k":
                    target = sample['target']
                    gold_answer_match = re.search(r'####\s*([^\n]+)', target)
                    gold_answer = gold_answer_match.group(1).strip() if gold_answer_match else None
                    question = sample['doc']['question']
                elif "gpqa" in task_name:
                    question = doc.get('Question')
                    target = doc.get('Correct Answer')
                    gold_answer = doc.get('answer')
                elif task_name in ["math500"] or "minerva" in task:
                    question = doc.get('problem')
                    target = doc.get('solution')
                    gold_answer = ground_truth
                elif "mbpp" in task_name:
                    question = doc.get('text', "")
                    target = doc.get('code', "")
                    test_list = doc.get('test_list', [])
                    gold_answer = "\n".join(test_list) if isinstance(test_list, list) else str(test_list)
                else:
                    question = str(doc)
                    target = "N/A"
                    gold_answer = "N/A"

                sample_data = {
                    'is_correct': int(is_correct_val),
                    'question': question,
                    'answer': target,
                    'ground_truth': gold_answer,
                    'extracted_answer': extracted if extracted else sample.get('filtered_resps', [None])[0],
                    'generation': generation
                }
                output_data['samples'][task].append(sample_data)
                
    return output_data