import os
import sys
import logging
logger = logging.getLogger(__name__)
import transformers
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    set_seed,
)
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
load_dotenv()
DATA_DIR = os.environ.get("DATA_DIR")
from data_modules import GSM8K, GSM8KForLLAMA


@hydra.main(version_base=None, config_path="exp_config/t5")
def main(cfg : DictConfig):
    parser = transformers.HfArgumentParser(TrainingArguments)
    trainer_args_dict = OmegaConf.to_container(cfg.trainer)
    training_args = parser.parse_dict(trainer_args_dict)[0]
    training_args.output_dir = os.path.join(DATA_DIR, training_args.output_dir)
    
    set_seed(training_args.seed)
    
    model_path = os.path.join(DATA_DIR, cfg.model.model_path)
    if 'llama' in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            assert tokenizer.unk_token is not None
            tokenizer.pad_token = tokenizer.unk_token
        model = AutoModelForCausalLM.from_pretrained(model_path, attention_dropout=0.1)
        gsm8k_module = GSM8KForLLAMA(cfg.data, tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        gsm8k_module = GSM8K(cfg.data, tokenizer)

    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        train_dataset=gsm8k_module.train_dataset,
        data_collator=gsm8k_module.data_collator,
    )
    trainer.train()
    if 'llama' in model_path:
        state_dict = trainer.model.state_dict()
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(training_args.output_dir, state_dict=cpu_state_dict) 
    else:
        trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()