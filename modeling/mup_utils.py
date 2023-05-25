# Copyright 2022 Microsoft Corporation.

from functools import partial
import math

from torch.utils.data import DataLoader

import seaborn as sns
import datasets
from transformers import default_data_collator

from modeling.lm_mup import MupGPT2Model

from torch.optim.lr_scheduler import LambdaLR

sns.set()

def get_dataloader(arch, signature_columns):
    # ****************************************************************************************************
    #                                       Load data
    # ****************************************************************************************************
    train_dataset = None
    final_lm_dir = "add-your-own-path"
    max_lm_train_samples = None
    if final_lm_dir is not None:
        print(f'From {final_lm_dir} / {max_lm_train_samples}')
        train_dataset = datasets.load_from_disk(final_lm_dir)
        if max_lm_train_samples is not None:
            train_dataset = train_dataset.select(range(max_lm_train_samples))
        print(f'{train_dataset}')

    assert train_dataset is not None

    lm_dataloader = DataLoader(
                train_dataset,
                collate_fn=default_data_collator,
                batch_size=8,
                num_workers=10,
                pin_memory=True,
            )

    return lm_dataloader


def get_lazy_model_from_scratch(config, mup=True,
                            readout_zero_init=True, query_zero_init=True, input_mult=1.0, width_mult_for_weights=1.0):
    def f():
        config_in = config
        model = MupGPT2Model._from_config(config_in)
        if mup:
            model._init_all_weights_for_mup(readout_zero_init, query_zero_init, input_mult, width_mult_for_weights)
        return model
    
    return f


def _get_linear_schedule_with_inverse_log_warmup_lr_lambda(current_step: int,
                                                            *, num_warmup_steps: int, num_training_steps: int):
    if current_step < num_warmup_steps:
        return math.log(current_step + 1) * 1.0 / math.log(float(max(1, num_warmup_steps)) + 1e-7)
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

def get_linear_schedule_with_inverse_log_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    lr_lambda = partial(
        _get_linear_schedule_with_inverse_log_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
