import os
import json
from tqdm import tqdm
import re
import numpy as np

from ruamel.yaml import YAML
import random
random.seed(0)

from dotenv import load_dotenv
load_dotenv()
DATA_DIR = os.environ.get("DATA_DIR")

from transformers import AutoTokenizer
import hydra
from omegaconf import DictConfig


def compare_similarity(string1, string2, t):
    data1, data2 = t.tokenize(string1), t.tokenize(string2)
    s1 = set(data1)
    s2 = set(data2)
    jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
    return jaccard


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


def process_rft_train(data_dir_lst, tokenizer, limit=0):
    sim_threshold = 0.7

    try:
        sft_train_file_first = f'model_outputs/{data_dir_lst[0]}/train/train_dpo_data.jsonl'
        assert os.path.exists(sft_train_file_first)
    except:
        exp_name = data_dir_lst[0].split('/')[0]
        data_dir_lst = [f'{exp_name}/sft-1', f'{exp_name}/sft-2', f'{exp_name}/sft-3']
        sft_train_file_first = f'model_outputs/{data_dir_lst[0]}/train/train_dpo_data.jsonl'

    lst = []
    with open(sft_train_file_first, 'r', encoding='utf-8') as read_file:
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

    cnt = 0
    no_gold_cnt = 0
    train_data_ls = []
    eval_data_ls = []
    sol_pre_question = []
    for item in tqdm(lst):

        origin_positives = item['positives']
        gold_solution = item['gold_ans']
        if len(gold_solution) > 1:
            if 't5' in data_dir_lst[0]:
                gold_solution = process_gold_solution(gold_solution)
            else:
                assert 'llama'in data_dir_lst[0]
            origin_positives.append(gold_solution)
        else:
            gold_solution = None
            no_gold_cnt += 1

        if len(origin_positives) == 0 and gold_solution is None:
            continue
            
        positives = list(set(origin_positives))
        positives.sort()

        positives_dedup = [gold_solution] if gold_solution else [positives[0]]
        if len(positives) > 1:
            for p in positives:
                indicator = True
                for p_n in positives_dedup:
                    sim_score = compare_similarity(p, p_n, tokenizer)
                    if sim_score > sim_threshold:
                        indicator = False
                        break
                if indicator:
                    positives_dedup.append(p)
            positives = positives_dedup

        p_ls = []
        max_threshold = 100 if limit == 0 else limit
        for p in positives[:max_threshold]:
            new_data = {
                "question": item["question"],
                "answer": p,
                "ans": 0,
            }
            p_ls.append(new_data)

        sol_pre_question.append(len(p_ls))
        train_data_ls.extend(p_ls)
        cnt += 1

    print(f'Total training data: {len(train_data_ls)}')
    print(f'Avg solutions per question: {np.mean(sol_pre_question):.2f}\n')
    if limit > 0:
        print(f'Set max thresold at {limit}')
        return train_data_ls
    else:
        print(f'90% percentile: {np.percentile(sol_pre_question, 90)}\n')
        return int(np.percentile(sol_pre_question, 90))


@hydra.main(version_base=None, config_path="../exp_config/t5")
def main(cfg : DictConfig):
    exp_name = cfg.exp_name
    run_name = cfg.trainer.run_name
    data_dir = os.path.join(exp_name, run_name)
    
    tokenizer_name = cfg.model.model_path
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(DATA_DIR, tokenizer_name))
    
    data_dir_lst = [f'{exp_name}/dpo-1', f'{exp_name}/dpo-2', f'{exp_name}/dpo-3']
    assert cfg.trainer.run_name.startswith('dpo')
    idx = int(cfg.trainer.run_name[-1])
    data_dir_lst = data_dir_lst[:idx][::-1]
    print(data_dir_lst)

    thres = process_rft_train(data_dir_lst, tokenizer)
    train_data = process_rft_train(data_dir_lst, tokenizer, thres)
    with open(f'model_outputs/{data_dir}/train/train_rft_processed.jsonl', 'w', encoding='utf-8') as file:
        for item in train_data:
            file.write(json.dumps(item) + '\n')

    if os.path.exists(f'results/{exp_name}.json'):
        with open(f'results/{exp_name}.json', 'r') as f:
            res = json.load(f)
    else:
        res = {}

    model_id = run_name
    if model_id not in res:
        res[model_id] = dict()
    res[model_id][f'generated_sft_data'] = len(train_data)

    with open(f'results/{exp_name}.json', 'w') as f:
        json.dump(res, f, indent=4)


if __name__ == "__main__":
    main()
