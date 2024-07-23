import json
import re
import os
from collections import Counter
from datasets import load_dataset
import hydra
from omegaconf import DictConfig, OmegaConf

ANS_RE = re.compile(r"#+ (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

def extract_answer(completion):
    # ans_lst = re.findall(r'\d*\.?\d+', completion)
    ans_lst = re.findall(ANS_RE, completion)
    if len(ans_lst) > 0:
        try:
            ans = re.sub(',', '', ans_lst[-1])
            ans = float(ans)
        except:
            ans = INVALID_ANS
    else:
        ans = INVALID_ANS
    return ans


def parse_gold(lines):
    all_ans = []
    for line in lines:
        # ans = extract_answer(line['answer'])
        all_ans.append(line['ans'])
    return all_ans


def parse(lines):
    all_ans = []
    for line in lines:
        ans = extract_answer(json.loads(line)[0][1])
        all_ans.append(ans)
    return all_ans

@hydra.main(version_base=None, config_path="exp_config/t5")
def eval_json(cfg : DictConfig):
    exp_name = cfg.exp_name
    run_name = cfg.trainer.run_name
    split = cfg.data.split
    json_path = os.path.join(
        'model_outputs/',
        exp_name,
        run_name, 
        split,
        f'greedy_decode.json',
        )
    with open(json_path, 'r') as f:
        lines = f.readlines()
    pred_ans = parse(lines)

    df = os.path.join('gsm8k', f'{split}.jsonl')
    gold_data = load_dataset('json', data_files=df)['train']
    lines = []
    for d in gold_data:
        lines.append(d)
    gold_ans = parse_gold(lines)

    cor = 0
    assert len(pred_ans) >= len(gold_ans)
    for i in range(len(gold_ans)):
        if pred_ans[i] != INVALID_ANS and abs(float(pred_ans[i]) - float(gold_ans[i])) < 1e-4:
            cor += 1
    print(f'#### {run_name}/{split}:')
    print(f'Acc: {cor}/{len(gold_ans)} = {cor/len(gold_ans) * 100:.1f}%')

    res = {}
    model_id = run_name
    if model_id not in res:
        res[model_id] = dict()
    res[model_id][f'{split}@1'] = f'{cor/len(gold_ans) * 100:.1f}'

    if not os.path.exists('results'):
        os.makedirs('results')

    if os.path.exists(f'results/{exp_name}.json'):
        with open(f'results/{exp_name}.json', 'r') as f:
            res = json.load(f)
    else:
        res = {}

    model_id = run_name
    if model_id not in res:
        res[model_id] = dict()
    res[model_id][f'{split}@1'] = f'{cor/len(gold_ans) * 100:.1f}'

    with open(f'results/{exp_name}.json', 'w') as f:
        json.dump(res, f, indent=4)

    return pred_ans


if __name__ == "__main__":
    # from jsonargparse import CLI
    # CLI(eval_json, as_positional=False)
    eval_json()
