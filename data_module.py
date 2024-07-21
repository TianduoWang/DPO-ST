import re
import os
import numpy as np
from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import DataCollatorForSeq2Seq, AutoTokenizer
from datasets import load_dataset
from dotenv import load_dotenv
load_dotenv()
DATA_DIR = os.environ.get("DATA_DIR")

class GSM8K:
    def __init__(self, cfg: DictConfig, tokenizer: AutoTokenizer):
        super().__init__()
        self.model_max_length = cfg.model_max_length
        self.max_src_len = cfg.max_src_len
        self.max_tgt_len = cfg.max_tgt_len
        self.tokenizer = tokenizer
        data_files = {'train': cfg.data_path}
        dataset_dict = load_dataset('json', data_files=data_files)
        column_names = dataset_dict['train'].column_names
        processed_dataset = dataset_dict.map(
            self.preprocess,
            batched=True,
            num_proc=16,
            remove_columns=column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on train dataset",
            )
        self.train_dataset = processed_dataset['train']

        data_files = {
            'train': cfg.train_data_path,
            'dev': cfg.dev_data_path,
            'test': cfg.test_data_path,
        }
        dataset_dict = load_dataset('json', data_files=data_files)
        processed_eval_dataset = dataset_dict.map(
            self.preprocess,
            batched=True,
            num_proc=16,
            remove_columns=column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on train dataset",
            )
        self.dataset = processed_eval_dataset

        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, padding=True,
            )

    @classmethod
    def process_question(cls, question):
        return question
    
    @classmethod
    def process_answer(cls, solution):
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

    def preprocess(self, examples):
        inputs, targets = [], []
        for i in range(len(examples['question'])):
            processed_question = self.process_question(examples['question'][i])
            inputs.append(processed_question)
            processed_answer = self.process_answer(examples['answer'][i])
            targets.append(processed_answer)

        model_inputs = self.tokenizer(
            inputs, max_length=self.max_src_len, padding=False, truncation=True, 
        )
        labels = self.tokenizer(
            text_target=targets, max_length=self.max_tgt_len, padding=False, truncation=True, 
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def pad(tensors: List[torch.Tensor], padding_value: int = 0, padding_side: str = "right") -> torch.Tensor:
    """
    Pads a list of tensors to the same shape along the first dimension.

    Args:
        tensors (`List[torch.Tensor]`):
            List of input tensors to pad.
        padding_value (`int`):
            Value to use for padding. Default is 0.
        padding_side (`str`):
            Side on which to add padding. Must be 'left' or 'right'. Default is 'right'.

    Returns:
        `torch.Tensor`:
            A single tensor containing the padded tensors.

    Examples:
        >>> import torch
        >>> pad([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
        tensor([[1, 2, 3],
                [4, 5, 0]])
        >>> pad([torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])])
        tensor([[[1, 2],
                [3, 4]],

                [[5, 6],
                [0, 0]]])
    """
    # Determine the maximum shape for each dimension
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    # Create an output tensor filled with the padding value
    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        # Determine the slice for the sequence dimension
        if padding_side == "left":
            seq_slice = slice(output_shape[0] - t.shape[0], output_shape[0])
        elif padding_side == "right":
            seq_slice = slice(0, t.shape[0])
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output

@dataclass
class DataCollatorForLLAMA:
    
    tokenizer: AutoTokenizer
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # first, pad everything to the same length
        padded_batch = {}
        for k in features[0].keys():
            # Set padding value based on the key
            if k.endswith("input_ids"):
                padding_value = self.tokenizer.pad_token_id
            elif k.endswith("labels"):
                padding_value = self.label_pad_token_id
            elif k.endswith("attention_mask"):
                padding_value = 0
            else:
                raise ValueError(f"Unexpected key in batch '{k}'")

            padding_side = self.tokenizer.padding_side
            to_pad = [torch.tensor(ex[k], dtype=torch.int64) for ex in features]
            padded_batch[k] = pad(to_pad, padding_value=padding_value, padding_side=padding_side)

        return padded_batch


class GSM8KForLLAMA:
    def __init__(self, cfg: DictConfig, tokenizer: AutoTokenizer):
        super().__init__()
        self.model_max_length = cfg.model_max_length
        self.max_src_len = cfg.max_src_len
        self.max_tgt_len = cfg.max_tgt_len
        self.tokenizer = tokenizer
        data_files = {'train': cfg.data_path}
        dataset_dict = load_dataset('json', data_files=data_files)
        column_names = dataset_dict['train'].column_names
        processed_dataset = dataset_dict.map(
            self.preprocess,
            batched=False,
            num_proc=16,
            remove_columns=column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on train dataset",
            )
        self.train_dataset = processed_dataset['train']

        eval_data_files = {'train': cfg.train_data_path, 'dev': cfg.dev_data_path, 'test': cfg.test_data_path}
        dataset_dict = load_dataset('json', data_files=eval_data_files)
        column_names = dataset_dict['dev'].column_names
        processed_eval_dataset = dataset_dict.map(
            self.preprocess_eval,
            batched=False,
            num_proc=16,
            remove_columns=column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on train dataset",
            )

        self.dataset = processed_eval_dataset
        self.data_collator = DataCollatorForLLAMA(tokenizer=tokenizer)

    @classmethod
    def process_question(cls, question):
        question_w_template = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{question}\n\n### Response:\n"
        )
        return question_w_template
    
    @classmethod
    def process_answer(cls, answer):
        return answer

    def build_tokenized_answer(self, prompt, answer):
        full_tokenized = self.tokenizer(prompt+answer, add_special_tokens=False)
        prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
        answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]
        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

        # Prepare input tokens for token by token comparison
        full_input_ids = np.array(full_tokenized["input_ids"])
        assert len(full_input_ids) == len(full_concat_input_ids)

        response_token_ids_start_idx = len(prompt_input_ids)
        # If tokenized prompt is different than both prompt+answer, then it means the
        # last token has changed due to merging.
        if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1
        
        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]
        assert len(prompt_input_ids) == len(prompt_attention_mask)

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

        return_dict = dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
        )
        return return_dict

    def preprocess(self, examples):
        batch = {}
        prompt = self.process_question(examples["question"])
        chosen = self.process_answer(examples["answer"])
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
        prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}
        chosen_tokens = self.build_tokenized_answer(prompt, chosen)
        prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])
        chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
        prompt_len_input_ids = chosen_prompt_len_input_ids
        for k, v in prompt_tokens.items():
            prompt_tokens[k] = v[:prompt_len_input_ids]
        
        bos_token_id = self.tokenizer.bos_token_id
        if prompt_len_input_ids == 0 or bos_token_id != prompt_tokens["prompt_input_ids"][0]:
            prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens["prompt_input_ids"]
            prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
        if chosen_prompt_len_input_ids == 0 or bos_token_id != chosen_tokens["prompt_input_ids"][0]:
            chosen_tokens["prompt_input_ids"] = [bos_token_id] + chosen_tokens["prompt_input_ids"]
            chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
        
        eos_token_id = self.tokenizer.eos_token_id
        if len(chosen_tokens["input_ids"]) == 0 or eos_token_id != chosen_tokens["input_ids"][-1]:
            chosen_tokens["input_ids"].append(eos_token_id)
            chosen_tokens["attention_mask"].append(1)

        # Truncate long sequences
        response_length = len(chosen_tokens["input_ids"])
        for answer_tokens in [chosen_tokens, prompt_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + response_length > self.model_max_length:
                for k in ["prompt_input_ids", "prompt_attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][-self.max_src_len :]

        # Create labels
        chosen_sequence_tokens = {
            k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
        }
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
            -100
        ] * len(chosen_tokens["prompt_input_ids"])

        for k, toks in {
            "chosen_": chosen_sequence_tokens,
            "": prompt_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}{type_key}"] = tokens
        new_batch = dict(
            input_ids=batch['chosen_input_ids'],
            attention_mask=batch['chosen_attention_mask'],
            labels=batch['chosen_labels'],
        )
        return new_batch

    def preprocess_eval(self, example):
        prompt = self.process_question(example["question"])
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
        batch = dict(
            input_ids=prompt_tokens['input_ids'],
            attention_mask=prompt_tokens['attention_mask'],
        )
        return batch