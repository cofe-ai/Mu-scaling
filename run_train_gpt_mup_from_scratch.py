import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
from transformers.deepspeed import is_deepspeed_zero3_enabled

import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
    GPT2Tokenizer
)
from transformers.trainer_utils import get_last_checkpoint

from transformers import GPT2Config
from modeling.initialize_with_mup import mup_init_from_scratch
from mup_trainer import MupTrainer
from utils import concat_path

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    model_load_pretrained: bool = field(
        default=False,
        metadata={"help": "Whether to load checkpoints even if `model_name_or_path` is provided"},
    )

    def __post_init__(self):
        if self.config_name is None and self.model_name_or_path is None:
            raise ValueError(
                "--config_name or --model_name_or_path must be set"
            )


@dataclass
class DataTrainingArguments:
    final_train_dir: Optional[str] = field(default=None, metadata={
        "help": "training data path"
    })
    final_lm_dir: Optional[str] = field(default=None, metadata={
        "help": "path to language model data"
    })
    final_tokenize_dir: Optional[str] = field(default=None, metadata={
        "help": "tokenizer path"
    })
    max_lm_train_samples: Optional[int] = field(default=None, metadata={
        "help": "maximum samples"
    })

    def __post_init__(self):
        if self.final_train_dir is not None:
            if self.final_lm_dir is None:
                self.final_lm_dir = concat_path(self.final_train_dir, 'lm')

            if self.final_tokenize_dir is None:
                self.final_tokenize_dir = concat_path(self.final_train_dir, 'tokenizer')


@dataclass
class MyTrainingArguments(TrainingArguments):
    hp_tune_base_width: Optional[int] = field(default=256, metadata={"help": "mup基础宽度 参数化时按照此宽度放缩"})
    size_per_head: Optional[int] = field(default=128, metadata={"help": "每个头的宽度 默认在参数化放缩中不变"})
    hp_tune_actual_width: Optional[int] = field(default=768, metadata={"help": "mup实际所调参的模型宽度"})
    output_mult: Optional[float] = field(default=1.0, metadata={"help": "输出层乘子，可微调超参数,当前方案中表示对ckpt的vocab除以该数值。"})
    initializer_range: Optional[float] = field(default=0.02, metadata={"help": "初始化标准差,覆盖config"})
    log_warmup: Optional[bool] = field(default=False, metadata={"help": "无deepspeed时是否使用log warmup"})
    unified_dropout: Optional[float] = field(default=None, metadata={"help": "若非none，将所有dropout层设为此值，主要用于零化dropout"})
    use_mup: Optional[bool] = field(default=True, metadata={"help": "mup开关默认打开，手动关闭用来跑对照实验"})
    exit_steps: Optional[int] = field(default=None, metadata={"help": "手动设定退出的step数"})
    readout_zero_init: Optional[bool] = field(default=True, metadata={"help": "vocab是否全零化"})
    query_zero_init: Optional[bool] = field(default=True, metadata={"help": "Q阵是否全零化"})
    is_training_ckpt_self: Optional[bool] = field(default=False, metadata={"help": "是否是加载checkpoint同大小正式训练"})


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # ****************************************************************************************************
    #                                       tokenizer
    # ****************************************************************************************************
    tokenizer = None
    # look for data directory first, then configuration
    if data_args.final_tokenize_dir is not None:
        logger.info(f'loading tokenizer from PREPROCESSED DATA: {data_args.final_tokenize_dir}')
        tokenizer = GPT2Tokenizer.from_pretrained(data_args.final_tokenize_dir, cache_dir=model_args.cache_dir)
    elif model_args.config_name is not None:
        logger.info(f'loading tokenizer from CONFIG: {model_args.config_name}')
        tokenizer = GPT2Tokenizer.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            f"FAILED to load tokenizer, please provide one of --final_tokenize_dir, --config_name"
        )

    # ****************************************************************************************************
    #                                       Load Config
    # ****************************************************************************************************
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision
    }
    if model_args.config_name:
        try:
            config = GPT2Config.from_pretrained(model_args.config_name, **config_kwargs)
        except OSError as e:
            if model_args.model_name_or_path:
                logger.warning(
                    f"failed to load from {model_args.config_name}. "
                    f"constructing with: {model_args.model_name_or_path} "
                )
                config_kwargs['vocab_size'] = len(tokenizer)
                config_kwargs['pad_token_id'] = tokenizer.pad_token_id
                config_kwargs['cls_token_id'] = tokenizer.cls_token_id
                config_kwargs['sep_token_id'] = tokenizer.sep_token_id
                config = GPT2Config.from_pretrained(model_args.model_name_or_path, **config_kwargs)

                config.save_pretrained(model_args.config_name)
            else:
                raise ValueError(
                    f"Must provide one of: --model_name_or_path, --num_multi_task_labels"
                )

    elif model_args.model_name_or_path :
        config_kwargs['vocab_size'] = len(tokenizer)
        config_kwargs['pad_token_id'] = tokenizer.pad_token_id
        config_kwargs['cls_token_id'] = tokenizer.cls_token_id
        config_kwargs['sep_token_id'] = tokenizer.sep_token_id
        config = GPT2Config.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        raise ValueError(
            f"--config_name, --model_name_or_path, --num_multi_task_labels"
        )

    # ****************************************************************************************************
    #                                       Model Init
    # ****************************************************************************************************

    ### Initialize Model with Mup ###
    training_args.width_mult_for_weights = (float(training_args.hp_tune_actual_width) / training_args.hp_tune_base_width) if training_args.use_mup else 1.0
    
    logger.info(f"width_mult_for_weights: {training_args.width_mult_for_weights}")
    config.width_mult_for_weights = training_args.width_mult_for_weights

    # Mup only supports training from scratch
    assert model_args.model_load_pretrained == False
    model = mup_init_from_scratch(config=config, training_args=training_args,
                                         model_args=model_args, logger=logger)

    #################################

    assert len(tokenizer) == model.config.vocab_size

    if is_deepspeed_zero3_enabled():
        n_params = 0
        n_partitioned_params = 0
        for p in model.parameters():
            if p.ds_tensor is not None:
                n_params += p.ds_numel
                n_partitioned_params += p.ds_tensor.numel()
        logger.info(
            f"My MSG: Training new model - Total size={n_params / 2 ** 20:.2f}M params. Total partitioned size={n_partitioned_params / 2 ** 20:.2f}M params ")
    else:
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        logger.info(f"My MSG: Training new model - Total size={n_params / 2 ** 20:.2f}M params")

    # ****************************************************************************************************
    #                                       Load Data
    # ****************************************************************************************************
    train_dataset = None
    if data_args.final_lm_dir is not None:
        logger.info(f'Loading LM data from: {data_args.final_lm_dir} / {data_args.max_lm_train_samples}')
        train_dataset = datasets.load_from_disk(data_args.final_lm_dir)
        if data_args.max_lm_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_lm_train_samples))
        logger.info(f'{train_dataset}')

    # ****************************************************************************************************
    #                                       Training Process
    # ****************************************************************************************************
    trainer = MupTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=None,
        preprocess_logits_for_metrics=None,
    )

    # Training
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    trainer.save_model()

    metrics = train_result.metrics
    max_lm_train_samples = (
        data_args.max_lm_train_samples if data_args.max_lm_train_samples is not None else len(train_dataset)
    )
    metrics["lm_train_samples"] = min(max_lm_train_samples, len(train_dataset))
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    kwargs = {
        "dataset": ', '.join([str(data_args.final_lm_dir), str(data_args.final_lm_dir)])
    }
    trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
