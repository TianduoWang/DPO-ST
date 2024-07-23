import json
import re
import os
from ruamel.yaml import YAML
from math import comb
from omegaconf import DictConfig, OmegaConf
import hydra

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
        all_ans.append(line['ans'])
    return all_ans


def parse(lines):
    all_ans = []
    for line in lines:
        ans = extract_answer(json.loads(line)[0][1])
        all_ans.append(ans)
    return all_ans


def get_gold_qa(split, max_data=0):

    with open(f'gsm8k/{split}.jsonl', 'r') as f:
        lines = f.readlines()

    ans_lst = []
    question_lst = []
    solution_lst = []
    for l in lines:
        data = json.loads(l)
        ans_lst.append(data['ans'])
        question_lst.append(data['question'])
        solution_lst.append(data['answer'])
        if max_data > 0 and len(ans_lst) == max_data:
            break
    return question_lst, solution_lst, ans_lst


# def get_gold_qa_train(filename, max_data=0):

#     with open(filename, 'r') as f:
#         lines = f.readlines()

#     ans_lst = []
#     question_lst = []
#     solution_lst = []
#     for l in lines:
#         data = json.loads(l)
#         ans_lst.append(data['ans'])
#         question_lst.append(data['question'])
#         solution_lst.append(data['answer'])
#         if max_data > 0 and len(ans_lst) == max_data:
#             break
#     return question_lst, solution_lst, ans_lst


def eval_json(json_path, gold_ans):

    with open(json_path, 'r') as f:
        lines = f.readlines()
    pred_ans = parse(lines)

    cor = 0
    assert len(pred_ans) >= len(gold_ans)
    for i in range(len(gold_ans)):
        if pred_ans[i] != INVALID_ANS and abs(float(pred_ans[i]) - float(gold_ans[i])) < 1e-4:
            cor += 1

    return {i:json.loads(lines[i])[0][1] for i in range(len(gold_ans))}, \
           {i:pred_ans[i] for i in range(len(gold_ans))}, \
           cor, \
           len(gold_ans)


@hydra.main(version_base=None, config_path="exp_config/t5")
# def eval_diverse(cfg:str, split:str, key:str, max_seed:int, t:float=0.7):
def eval_diverse(cfg : DictConfig):
    # yaml=YAML(typ='safe')
    # with open(cfg, 'r', encoding='utf-8') as f:
    #     load_dict = yaml.load(f)
    #     exp_name = load_dict['general']['exp_name']
    #     args_dict = load_dict[key]
    # run_name = args_dict['run_name']

    exp_name = cfg.exp_name
    run_name = cfg.trainer.run_name
    split = cfg.data.split
    temperature = cfg.eval.sampling.temperature
    json_path = os.path.join(
        'model_outputs/', 
        exp_name,
        run_name, 
        split,
        )
    print(json_path)

    max_seed = cfg.eval.sampling.max_seed
    path_list = [os.path.join(json_path, f'seed_{idx}-t_{temperature}.json') for idx in range(0,max_seed,1)]
    path_list = [f for f in path_list if os.path.exists(f)]
    assert len(path_list) == max_seed

    # if split == 'train':
    #     questions, solutions, gold_ans = get_gold_qa_train(load_dict['general']['origin_train_data'])
    # else:
    #     questions, solutions, gold_ans = get_gold_qa(split)
    questions, solutions, gold_ans = get_gold_qa(split)

    all_q = []
    all_ans = []
    new_path_list = []
    for file_path_idx in range(len(path_list)):
        file_path = path_list[file_path_idx]
        res, pred, _, cnt = eval_json(file_path, gold_ans)
        all_q.append(res)
        all_ans.append(pred)
        new_path_list.append(file_path)
    path_list = new_path_list

    output = [{} for _ in range(len(gold_ans))]
    for i in range(len(output)):
        output[i]['question'] = questions[i]
        output[i]['gold_ans'] = solutions[i]
        output[i]['positives'] = []
        output[i]['negatives'] = []

    for file_path_idx in range(len(path_list)):
        for idx in range(len(gold_ans)):
            if all_ans[file_path_idx][idx] != INVALID_ANS and float(all_ans[file_path_idx][idx]) == float(gold_ans[idx]):
                key = 'positives'
            else:
                key = 'negatives'
            solution = all_q[file_path_idx][idx]
            output[idx][key].append(solution)

    no_positives = 0
    no_negatives = 0
    with open(f'{json_path}/{split}_dpo_data.jsonl', 'w') as f:
        for item in output:
            if len(item['positives']) == 0:
                no_positives += 1
            if len(item['negatives']) == 0:
                no_negatives += 1

            f.write(json.dumps(item) + '\n')

    if split == 'train':
        print(f'# w/o positives: {no_positives} ({no_positives/len(gold_ans)*100:.1f}%)')
        print(f'# w/o negatives: {no_negatives} ({no_negatives/len(gold_ans)*100:.1f}%)')
    corrects = len(gold_ans) - no_positives
    pass_at_10 = corrects/len(gold_ans)
    print(f'\nPass 1@{max_seed}: {corrects} / {len(gold_ans)} = {pass_at_10*100:.1f}')


    if max_seed >= 5:
        pass_5_corr = 0
        for item in output:
            totay_ways = comb(max_seed, 5)
            all_wrong_ways = comb(len(item['negatives']), 5)
            pass_5_corr += 1 - all_wrong_ways / totay_ways
        pass_at_5 = pass_5_corr/len(gold_ans)
        print(f'\nPass 1@5: {int(pass_5_corr)} / {len(gold_ans)} = {pass_at_5*100:.1f}')

        pass_3_corr = 0
        for item in output:
            totay_ways = comb(max_seed, 3)
            all_wrong_ways = comb(len(item['negatives']), 3)
            pass_3_corr += 1 - all_wrong_ways / totay_ways
        pass_at_3 = pass_3_corr/len(gold_ans)
        print(f'\nPass 1@3: {int(pass_3_corr)} / {len(gold_ans)} = {pass_at_3*100:.1f}')

    res = {}
    model_id = run_name
    if model_id not in res:
        res[model_id] = dict()
    res[model_id][f'{split}@{max_seed}'] = f'{pass_at_10 * 100:.1f}'
    if max_seed >= 5:
        res[model_id][f'{split}@5'] = f'{pass_at_5 * 100:.1f}'
        res[model_id][f'{split}@3'] = f'{pass_at_3 * 100:.1f}'

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
    res[model_id][f'{split}@{max_seed}'] = f'{pass_at_10 * 100:.1f}'
    if max_seed >= 5:
        res[model_id][f'{split}@5'] = f'{pass_at_5 * 100:.1f}'
        res[model_id][f'{split}@3'] = f'{pass_at_3 * 100:.1f}'

    with open(f'results/{exp_name}.json', 'w') as f:
        json.dump(res, f, indent=4)
    print('-=-=-=-=-=-=-=-=-=')



if __name__ == "__main__":
    # from jsonargparse import CLI
    # CLI(eval_diverse, as_positional=False)
    eval_diverse()