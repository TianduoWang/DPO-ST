import os
import json
from tqdm import tqdm
from ruamel.yaml import YAML
import random
import re
random.seed(0)

from dotenv import load_dotenv
load_dotenv()
DATA_DIR = os.environ.get("DATA_DIR")

from transformers import AutoTokenizer
import hydra
from omegaconf import DictConfig, OmegaConf
from data_modules import GSM8K, GSM8KForLLAMA


def compare_similarity(string1, string2, t):
    data1, data2 = t.tokenize(string1), t.tokenize(string2)
    s1 = set(data1)
    s2 = set(data2)
    actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
    return actual_jaccard

def process_gold_solution(solution):
    solution = re.sub(r'<<\d+=\d+>>', '', solution)
    solution = re.sub(r'(\D):(\d)', r'\1: \2', solution)
    solution = re.sub(r'<<', '[ ', solution)
    solution = re.sub(r'>>', '] ', solution)
    solution = re.sub(r'\$([^\s])', r'$ \1', solution)
    solution = re.sub(r'([^\s])\+([^\s])', r'\1 + \2', solution)
    solution = re.sub(r'([^\s])-([^\s])', r'\1 - \2', solution)
    solution = re.sub(r'([^\s])\*([^\s])', r'\1 * \2', solution)
    solution = re.sub(r'([^\s])/([^\s])', r'\1 / \2', solution)
    solution = re.sub(r'([^\s])=([^\s])', r'\1 = \2', solution)
    
    #----------------------
    solution_split_by_line = solution.split('\n')
    sol = []
    for l in solution_split_by_line:
        if l.startswith('##'):
            sol.append(l)
        elif not l.endswith('.'):
            sol.append(l+'.')
        else:
            sol.append(l)
    solution = ' '.join(sol)
    return solution


def process_dpo_train(data_dir_lst, tokenizer, process_question_func, process_answer_func):
    sim_threshold = 0.7

    dpo_train_file_first = f'model_outputs/{data_dir_lst[0]}/train/train_dpo_data.jsonl'
    assert os.path.exists(dpo_train_file_first)
    lst = []
    with open(dpo_train_file_first, 'r', encoding='utf-8') as read_file:
        dataset_first = read_file.readlines()
        for d in dataset_first:
            lst.append(eval(d))

    for data_dir in data_dir_lst[1:]:
        dpo_train_file = f'model_outputs/{data_dir}/train/train_dpo_data.jsonl'
        if os.path.exists(dpo_train_file):
            with open(dpo_train_file, 'r', encoding='utf-8') as read_file:
                dataset = read_file.readlines()
                assert len(dataset) == len(lst)
                for idx, d in enumerate(dataset):
                    lst[idx]['positives'].extend(eval(d)['positives'])
                    lst[idx]['negatives'].extend(eval(d)['negatives'])

    cnt = 0
    no_pos_cnt = 0
    no_neg_cnt = 0
    train_data_ls = []
    eval_data_ls = []
    instances_per_question = []
    for item in tqdm(lst):

        if len(item['negatives']) == 0:
            no_neg_cnt += 1
            instances_per_question.append(0)
            continue
        
        # if len(item['positives']) == 0:
        #     no_pos_cnt += 1
        #     instances_per_question.append(0)
        #     continue
        #     gold_rationale = item['gold_ans']
        #     gold_rationale = process_gold_solution(gold_rationale)
        #     item['positives'].append(gold_rationale)

        positives = list(set(item['positives']))
        negatives = list(set(item['negatives']))
        positives.sort()
        negatives.sort()

        gold_rationale = item['gold_ans']
        # gold_rationale = process_gold_solution(gold_rationale)
        gold_rationale = process_answer_func(gold_rationale)
        positives = [gold_rationale] + positives

        positives_dedup = [positives[0]]
        if len(positives) > 1:
            for p in positives[1:]:
                indicator = True
                for p_n in positives_dedup:
                    if compare_similarity(p, p_n, tokenizer) > sim_threshold:
                        indicator = False
                        break
                if indicator:
                    positives_dedup.append(p)
            positives = positives_dedup

        negatives_dedup = [negatives[0]]
        if len(negatives) > 1:
            for p in negatives[1:]:
                indicator = True
                for p_n in negatives_dedup:
                    if compare_similarity(p, p_n, tokenizer) > sim_threshold:
                        indicator = False
                        break
                if indicator:
                    negatives_dedup.append(p)
        negatives = negatives_dedup

        random.shuffle(positives)
        random.shuffle(negatives)

        p_ls = []
        for positive in positives:
            for negative in negatives:
                sim_score = compare_similarity(positive, negative, tokenizer)
                new_data = ({
                    "prompt": process_question_func(item["question"]),
                    "chosen": positive,
                    "rejected": negative
                }, sim_score)
                p_ls.append(new_data)
        p_ls.sort(key=lambda item: item[1])
        p_train_data = [item[0] for item in p_ls[:3]]
        train_data_ls.extend(p_train_data)
        instances_per_question.append(len(p_train_data))
        cnt += 1
    print(f'Processed num questions: {cnt}')
    print(f'Total training data: {len(train_data_ls)}')
    return train_data_ls



@hydra.main(version_base=None, config_path="../exp_config/t5")
# def main(cfg:str, key:str, tokenizer_name:str):
def main(cfg : DictConfig):
    exp_name = cfg.exp_name
    run_name = cfg.trainer.run_name
    data_dir = os.path.join(exp_name, run_name)

    tokenizer_name = cfg.model.model_path
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(DATA_DIR, tokenizer_name))

    data_dir_lst = [f'{exp_name}/sft-0', f'{exp_name}/sft-1', f'{exp_name}/sft-2']
    assert cfg.trainer.run_name.startswith('sft')
    idx = int(cfg.trainer.run_name[-1])+1
    data_dir_lst = data_dir_lst[:idx][::-1]

    if 'llama' in exp_name:
        process_question_func = GSM8KForLLAMA.process_question
        process_answer_func = GSM8KForLLAMA.process_answer
    else:
        assert 't5' in exp_name
        process_question_func = GSM8K.process_question
        process_answer_func = GSM8K.process_answer
    
    train_data = process_dpo_train(data_dir_lst, tokenizer, process_question_func, process_answer_func)
    with open(f'model_outputs/{data_dir}/train/train_dpo_processed.jsonl', 'w', encoding='utf-8') as file:
        for item in train_data:
            file.write(json.dumps(item) + '\n')

    eval_data = train_data[:320]
    with open(f'model_outputs/{data_dir}/train/eval_dpo_processed.jsonl', 'w', encoding='utf-8') as file:
        for item in eval_data:
            file.write(json.dumps(item) + '\n')

    if os.path.exists(f'results/{exp_name}.json'):
        with open(f'results/{exp_name}.json', 'r') as f:
            res = json.load(f)
    else:
        res = {}

    model_id = run_name
    if model_id not in res:
        res[model_id] = dict()
    res[model_id][f'generated_dpo_data'] = len(train_data)

    with open(f'results/{exp_name}.json', 'w') as f:
        json.dump(res, f, indent=4)


if __name__ == "__main__":
    main()
