from transformers import pipeline
from datasets import load_dataset, load_metric
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import polars as pl

import torch

model_ckpt = "facebook/bart-large-cnn"
# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)
device = "cuda" if torch.cuda.is_available() else "cpu"
dataset_samsum = load_dataset("samsum")
print(dataset_samsum)
dataset_cnn = load_dataset("cnn_dailymail", "3.0.0")
print(dataset_cnn)
pipe = pipeline('summarization', model = model_ckpt )
pipe_out = pipe(dataset_samsum['test'][0]['dialogue'] )
print(dataset_samsum['test'][0]['dialogue'])
print(dataset_cnn['train'][0]['highlights'])
pipe_out = pipe(dataset_cnn['train'][0]['article'])
print(pipe_out[0]['summary_text'])
