import logging
import torch
from downstream import SequenceClassification
from transformers import BertConfig, BertTokenizer, BertModel
from transformers import GlueDataset

from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    #glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)


class GlueDataArgs:
    def __init__(self, task_name, data_dir, max_length=128):
        self.task_name=task_name.lower()
        self.data_dir=data_dir
        self.max_seq_length=max_length
        self.overwrite_cache=False
        
        
bert_path="/GPUFS/nsccgz_xliao_djs/bert_multiend/bert-models/bert-base-cased"

config = BertConfig.from_pretrained(
        bert_path,
        num_labels=2,
        finetuning_task="sst-2",
        cache_dir="/GPUFS/nsccgz_xliao_djs/bert_multiend/pre-training-multi-task/output",
    )       
        
model_task0 = SequenceClassification(config)


config1 = BertConfig.from_pretrained(
        bert_path,
        num_labels=2,
        finetuning_task="qqp",
        cache_dir="/GPUFS/nsccgz_xliao_djs/bert_multiend/pre-training-multi-task/output",
    )

model_task1 = SequenceClassification(config1)
print(model_task0)
print(model_task1)

print(config1)

parser = HfArgumentParser(TrainingArguments)
training_args = parser.parse_args_into_dataclasses()
print(training_args)
