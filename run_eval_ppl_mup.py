import argparse
import torch
from torch.nn import CrossEntropyLoss
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer)
from modeling.lm_mup import MupGPT2Model
from mup import set_base_shapes

parser = argparse.ArgumentParser()
# Load Model
parser.add_argument("--model_name_or_path", type=str, default='gpt2')
parser.add_argument("--is_ours", default=False, action='store_true')

# Dataset
parser.add_argument("--dataset_path", type=str, default=None)
parser.add_argument("--dataset_name", type=str, default=None)
parser.add_argument("--dataset_split_name", type=str, default=None)
parser.add_argument("--dataset_feature_name", type=str, default='text')
parser.add_argument("--is_disk_data", default=False, action='store_true')

# Others
parser.add_argument("--stride", type=int, default=None)
parser.add_argument("--cache_dir", type=str, default=None)
parser.add_argument("--data_dir", type=str, default=None)
params = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

class cheat_infshape(object):
    def __init__(self, width_mult):
        self.width_mult_ = width_mult
    def width_mult(self):
        return self.width_mult_


# Load Model
model = None
tokenizer = None
if params.is_ours:
    model = MupGPT2Model.from_pretrained(params.model_name_or_path).to(device)
    model.transformer.input_mult = model.config.output_mult
    model.lm_head.weight.infshape = cheat_infshape(model.config.n_embd / 256)
    print(f"self.lm_head.output_mult:{model.lm_head.output_mult}")
    print(f"self.transformer.input_mult:{model.transformer.input_mult}")
    print(f"self.lm.width_mult:{model.lm_head.width_mult()}")
    tokenizer = GPT2Tokenizer.from_pretrained(params.model_name_or_path)
    
else:
    model = AutoModelForCausalLM.from_pretrained(params.model_name_or_path, cache_dir="/share/project/lixiang/cache").to(device)
    tokenizer = AutoTokenizer.from_pretrained(params.model_name_or_path, cache_dir="/share/project/lixiang/cache")
assert model is not None and tokenizer is not None

# Load Data
raw_datasets = None
if params.is_disk_data:
    raw_datasets = DatasetDict.load_from_disk(params.dataset_path)
else:
    raw_datasets = load_dataset(params.dataset_path, params.dataset_name,
                                cache_dir=params.cache_dir,
                                data_dir=params.cache_dir
                                )
assert raw_datasets is not None

# Preprocessing
target_dataset = None
if params.dataset_split_name is not None:
    target_dataset = params.dataset_split_name
elif "test" in raw_datasets.keys():
    target_dataset = "test"
elif "validation" in raw_datasets.keys():
    target_dataset = "validation"
elif "train" in raw_datasets.keys():
    target_dataset = "train"
assert target_dataset is not None
target_dataset = "train"
print(f'dataset: {params.dataset_path} - {params.dataset_name} - {target_dataset}')
dataset_test = raw_datasets[target_dataset]
encodings = tokenizer("\n\n".join(dataset_test[params.dataset_feature_name]), return_tensors="pt")

# Eval
ignore_index = CrossEntropyLoss().ignore_index
if params.model_name_or_path.startswith('xlnet'):
    max_length = 1024
else:
    max_length = model.config.n_positions
# stride = max_length // 2
stride = params.stride if params.stride is not None else max_length
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    attention_mask = encodings.attention_mask[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = ignore_index

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)
        neg_log_likelihood = outputs.loss * trg_len

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

# ppl = torch.exp(torch.stack(nlls).sum() / seq_len)
ppl = torch.stack(nlls).sum() / seq_len
n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
print(f"model = {n_params / 2 ** 20:.2f}M params; ppl = {ppl}")
