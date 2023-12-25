import json
import os
import sys

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np

from utils import load_data

np.random.seed(42)
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

model_id = "Salesforce/codet5p-220m-bimodal"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(model_id,  trust_remote_code=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def process_context_in_batches(context, batch_size=20):
    context_batches = [context[i:i + batch_size] for i in range(0, len(context), batch_size)]
    processed_context = []
    for batch in context_batches:
        processed_context.extend(code_to_nl_batch(batch))
    
    return processed_context



def code_to_nl_batch(codes: list[str], max_length: int = 100) -> list[str]:
    encoded_input = tokenizer(codes, return_tensors="pt", padding=True, truncation=True).to(device)
    input_ids = encoded_input.input_ids
    generated_ids = model.generate(input_ids, max_length=max_length).cpu()
    return [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]

for subset in ['easy', 'hard']:

    settings = 'cross_file_first'
    data = load_data('test', 'r', 'python', settings)
    raw_samples = data[subset]

    samples = []
    for i, sample in tqdm(enumerate(raw_samples), total=len(raw_samples)):
        sample['nl_code'] = code_to_nl_batch([sample['code']])[0]
        sample['nl_next_line'] = code_to_nl_batch([sample['next_line']])[0]
        
        # sample['nl_context'] = code_to_nl_batch(sample['context'])
        sample['nl_context'] = process_context_in_batches(sample['context'])
        
        samples.append(sample)

    # Convert the samples list to JSON Lines format and save to a file

    jsonl_data = "\n".join(json.dumps(sample) for sample in samples)

    jsonl_file_path = f"data/codet5p_{subset}_samples.jsonl"

    with open(jsonl_file_path, "w") as file:
        file.write(jsonl_data)

