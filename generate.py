import os
import sys
import argparse
import re
import torch
from torch.utils.data.distributed import DistributedSampler
import transformers
from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    GenerationConfig,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizerBase,
    load_tool,
    LogitsProcessor,
    set_seed,
    )
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Sequence, List, Any, Union
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import json

from tqdm import tqdm
import copy
from torch.cuda.amp import autocast
from datasets import load_dataset, concatenate_datasets
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()
from ruamel.yaml import YAML
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
load_dotenv()
DATA_DIR = os.environ.get("DATA_DIR")
from data_modules import GSM8K, GSM8KForLLAMA


def sequence_gather(s, world_size, pad_tok_id):
    local_size = torch.tensor(s.size(), device=s.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_length = max(size[1] for size in all_sizes)
    length_diff = max_length.item() - local_size[1].item()
    if length_diff:
        pad_size = (s.shape[0], length_diff)
        padding = torch.ones(pad_size, device=s.device, dtype=s.dtype)*pad_tok_id
        s = torch.concat((s, padding), dim = -1)
    gathered_s = [torch.ones_like(s)*pad_tok_id for _ in range(world_size)]
    dist.all_gather(gathered_s, s)
    return gathered_s


def tool_fn(expr):
    raw_result = eval(expr)
    f_res = float(raw_result)
    if abs(int(f_res)-f_res) < 0.0001:
        return str(int(f_res))
    else:
        return str(round(f_res, 2))


class CustomLogitsProcessor(LogitsProcessor):

    def __init__(self, tokenizer, tool):
        self.tokenizer = tokenizer
        self.tool = tool
        self.eq_token_id = tokenizer.convert_tokens_to_ids('▁=')
        self.right_token_id = tokenizer.convert_tokens_to_ids(']')
        self.left_token_id = tokenizer.convert_tokens_to_ids('▁[')

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        bz = scores.size(0)
        for b in range(bz):
            if input_ids[b][-1].item() == self.tokenizer.pad_token_id:
                continue
            if input_ids[b][-1].item() == self.tokenizer.eos_token_id:
                continue
            cur_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[b])
            meet_right = False
            meet_eq = None
            for i in range(len(cur_tokens)-1, -1, -1):
                if cur_tokens[i] == ']':
                    break
                elif cur_tokens[i] == '▁=':
                    meet_eq = i
                elif cur_tokens[i] == '▁[' and meet_eq is not None:
                    string = ''.join(cur_tokens[i+1:meet_eq])
                    string = re.sub('▁', ' ', string)
                    try:
                        res = self.tool(string)
                    except:
                        break
                    target_token_ids = self.tokenizer.encode(res, add_special_tokens=False, return_tensors='pt')[0]
                    if all(torch.eq(target_token_ids, input_ids[b][-len(target_token_ids):].cpu())):
                        scores[b, self.right_token_id] = 10e7
                    elif input_ids[b][-1] == self.eq_token_id:
                        scores[b, target_token_ids[0]] = 10e7
                    else:
                        gen_token_ids = input_ids[b][meet_eq+1:]
                        l = len(gen_token_ids)
                        scores[b, target_token_ids[l]] = 10e7
                    break
        return scores

class CustomLogitsProcessorLLAMA(LogitsProcessor):

    def __init__(self, tokenizer, tool):
        self.tokenizer = tokenizer
        self.tool = tool
        self.right_token_id = tokenizer.convert_tokens_to_ids('>>')
        self.space_token_id = tokenizer.convert_tokens_to_ids('▁')

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        bz = scores.size(0)
        
        for b in range(bz):
            if input_ids[b][-1].item() == self.tokenizer.pad_token_id:
                continue
            if input_ids[b][-1].item() == self.tokenizer.eos_token_id:
                continue
            # 查找有无闭合的 <<...>>
            # current_string = self.tokenizer.decode(input_ids[b])
            current_string = self.tokenizer.decode(input_ids[b][100:])
            pattern = r'<<[^>]+=[^>]*$'
            match = re.search(pattern, current_string)

            if match:
                focus_string = match.group(0).strip('<<')
                try:
                    eq_left, eq_right = focus_string.split('=')
                    num_result = self.tool(eq_left)
                    if float(num_result) < 0:
                        continue
                except:
                    continue
                
                if num_result == eq_right:
                     scores[b, self.right_token_id] = 1e6
                else:
                    res_token_ids = self.tokenizer.encode(num_result, add_special_tokens=False)
                    if res_token_ids[0] == self.space_token_id:
                        res_token_ids = res_token_ids[1:]
                    res_token_ids = torch.LongTensor(res_token_ids)
                    if len(eq_right) == 0:
                        scores[b, res_token_ids[0]] = 1e6
                    else:
                        try:
                            scores[b, res_token_ids[len(eq_right)]] = 1e6
                        except:
                            continue
        return scores


@hydra.main(version_base=None, config_path="exp_config/t5")
def main(cfg : DictConfig):

    exp_name = cfg.exp_name
    run_name = cfg.trainer.run_name
    split = cfg.data.split
    bz = cfg.eval.per_device_eval_batch_size

    rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    torch.manual_seed(cfg.trainer.seed)
    world_size = torch.cuda.device_count()

    base_model = os.path.join(DATA_DIR, cfg.trainer.output_dir)
    if 'llama' in base_model:
        tokenizer = AutoTokenizer.from_pretrained(base_model, truncation_side='left', padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
        gsm8k_module = GSM8KForLLAMA(cfg.data, tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        gsm8k_module = GSM8K(cfg.data, tokenizer)
    assert tokenizer.pad_token is not None

    torch.cuda.set_device(rank)
    model.to(torch.cuda.current_device())
    model = DDP(model, device_ids=[torch.cuda.current_device()])
    model.eval()

    eval_dataset = gsm8k_module.dataset[split]
    effective_bz = world_size * bz
    if len(eval_dataset) % effective_bz != 0:
        diff = effective_bz - len(eval_dataset) % effective_bz
        for_pad = eval_dataset.select(list(range(diff+20)))
        eval_dataset = concatenate_datasets([eval_dataset, for_pad])

    sampler = DistributedSampler(
        eval_dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False,
        )
    dataloader = DataLoader(
        eval_dataset, 
        shuffle=False, 
        collate_fn=gsm8k_module.data_collator, 
        batch_size=bz,
        sampler=sampler,
        drop_last=True,
        num_workers=12,
    )

    if cfg.eval.mode == 'greedy':
        out_path = os.path.join(
            'model_outputs/', exp_name, run_name, split,
            'greedy_decode.json',
            )
        generation_config = GenerationConfig(
            do_sample=False,
            max_new_tokens=cfg.eval.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        if cfg.eval.use_calc:
            m = CustomLogitsProcessorLLAMA(tokenizer=tokenizer, tool=tool_fn) if 'llama' in base_model \
                else CustomLogitsProcessor(tokenizer=tokenizer, tool=tool_fn)
            logits_proc = [m]
        else:
            logits_proc = []

        if rank == 0:
            iterator = tqdm(enumerate(dataloader), total=len(eval_dataset)//(bz*world_size))
        else:
            iterator = enumerate(dataloader)
        all_outputs = []
        for _, batch in iterator:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    output_ids = model.module.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        generation_config=generation_config,
                        logits_processor=logits_proc,
                        return_dict_in_generate=False,
                    )
            gather_outputs  = sequence_gather(output_ids, world_size, tokenizer.pad_token_id)
            gathered_inputs = sequence_gather(input_ids, world_size, tokenizer.pad_token_id)
            
            gather_outputs  = torch.stack(gather_outputs)
            gathered_inputs = torch.stack(gathered_inputs)
            
            gather_outputs  = gather_outputs.transpose(0,1).reshape(bz*world_size, -1)
            gathered_inputs = gathered_inputs.transpose(0,1).reshape(bz*world_size,-1)
            
            outputs_string = tokenizer.batch_decode(gather_outputs, skip_special_tokens=True)
            inputs_string  = tokenizer.batch_decode(gathered_inputs, skip_special_tokens=True)
            
            for idx in range(len(inputs_string)):
                temp = [[inputs_string[idx], outputs_string[idx].replace(inputs_string[idx], '')]]
                all_outputs.append(temp)
        
        if rank == 0:
            folder = '/'.join(out_path.split('/')[:-1])
            if not os.path.exists(folder):
                os.makedirs(folder)
            with open(out_path, 'w') as f:
                for item in all_outputs:
                    f.write(json.dumps(item) + '\n')
        dist.barrier()
    else:
        assert cfg.eval.mode == 'sampling'
        out_path = os.path.join(
            'model_outputs/', exp_name, run_name, f'{split}/',
            )
        generation_config = GenerationConfig(
            do_sample=True,
            max_new_tokens=300,
            temperature=cfg.eval.sampling.temperature,
        )

        m = CustomLogitsProcessor(
            tokenizer=tokenizer, 
            tool=tool_fn, 
        )

        if rank == 0:
            pbar = tqdm(total=(cfg.eval.sampling.max_seed - cfg.eval.sampling.min_seed) * len(eval_dataset)//effective_bz)

        for seed in range(cfg.eval.sampling.min_seed, cfg.eval.sampling.max_seed):
            set_seed(seed)
            all_outputs = []
            for _, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        output_ids = model.module.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            generation_config=generation_config,
                            logits_processor=[m],
                            return_dict_in_generate=False,
                        )
                gather_outputs = sequence_gather(output_ids, world_size, tokenizer.pad_token_id)
                gathered_inputs = sequence_gather(input_ids, world_size, tokenizer.pad_token_id)
                
                gather_outputs = torch.stack(gather_outputs)
                gathered_inputs = torch.stack(gathered_inputs)
                
                gather_outputs = gather_outputs.transpose(0,1).reshape(effective_bz, -1)
                gathered_inputs = gathered_inputs.transpose(0,1).reshape(effective_bz,-1)
                
                outputs_string = tokenizer.batch_decode(gather_outputs, skip_special_tokens=True)
                inputs_string = tokenizer.batch_decode(gathered_inputs, skip_special_tokens=True)
                
                for idx in range(len(inputs_string)):
                    temp = [[inputs_string[idx], outputs_string[idx].replace(inputs_string[idx], '')]]
                    all_outputs.append(temp)
                
                if rank == 0:
                    pbar.update(1)

            
            if rank == 0:
                folder = '/'.join(out_path.split('/')[:-1])
                if not os.path.exists(folder):
                    os.makedirs(folder)
                with open(out_path+f'seed_{seed}-t_{cfg.eval.sampling.temperature}.json', 'w') as f:
                    for item in all_outputs:
                        f.write(json.dumps(item) + '\n')

            dist.barrier()

        if rank == 0:
            pbar.close()
    return

if __name__ == "__main__":
    main()
