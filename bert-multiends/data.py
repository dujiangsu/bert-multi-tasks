from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from datasets import load_dataset, load_metric, load_from_disk
import random
import logging
import torch
from sklearn.metrics import f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
from transformers import default_data_collator

# use dataset in disk rather than from the internet
dataset_dir = "/home/nsccgz_jiangsu/glue-datasets"

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


# max_length is a hyperparameter
class GlueDataArgs:
    def __init__(self, task_name, max_length=128, pad_to_max_length=True):
        self.task_name=task_name.lower()
        self.max_seq_length=max_length
        self.overwrite_cache=False
        self.pad_to_max_length=pad_to_max_length

        
class DataIterator(object):
    def __init__(self, data_args, tokenizer, mode, cache_dir, batch_size):
    
        logger = logging.getLogger(__name__)
        if data_args.task_name == "sst-2":
            data_args.task_name = "sst2"
            
        if data_args.task_name is not None:
            self.datasets = load_from_disk(dataset_dir + "/" + data_args.task_name)            
            # load_dataset(path = "/datasets/glue/glue.py", name = data_args.task_name)
            
        if data_args.task_name is not None:
            is_regression = data_args.task_name == "stsb"
            if not is_regression:
                label_list = self.datasets["train"].features["label"].names
                num_labels = len(label_list)
            else:
                num_labels = 1
        else:
            # Trying to have good defaults here, don't hesitate to tweak to your needs.
            is_regression = self.datasets["train"].features["label"].dtype in ["float32", "float64"]
            if is_regression:
                num_labels = 1
            else:
                label_list = self.datasets["train"].unique("label")
                label_list.sort()  # Let's sort it for determinism
                num_labels = len(label_list)
        
        # Preprocessing the datasets
        if data_args.task_name is not None:
            sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
        else:
            # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
            non_label_column_names = [name for name in self.datasets["train"].column_names if name != "label"]
            if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
                sentence1_key, sentence2_key = "sentence1", "sentence2"
            else:
                if len(non_label_column_names) >= 2:
                    sentence1_key, sentence2_key = non_label_column_names[:2]
                else:
                    sentence1_key, sentence2_key = non_label_column_names[0], None    
            
        # Padding strategy
        if data_args.pad_to_max_length == True:
            padding = "max_length"
            max_length = data_args.max_seq_length
        # else: # TODO
   
        
        # Some models have set the order of the labels to use, so let's make sure we do use it.
        label_to_id = None
           

        def preprocess_function(examples):
            # Tokenize the texts
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer(*args, padding=padding, max_length=max_length, truncation=True)

            # Map labels to IDs (not necessary for GLUE tasks)
            if label_to_id is not None and "label" in examples:
                result["label"] = [label_to_id[l] for l in examples["label"]]
            return result
        
        self.datasets = self.datasets.map(preprocess_function, batched=True, load_from_cache_file = not data_args.overwrite_cache)
        
        if(mode == "train"):
            self.datasets = self.datasets["train"]
            self.sampler = RandomSampler(self.datasets)
        elif(mode == "dev"):
            self.datasets = self.datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
            self.sampler = SequentialSampler(self.datasets)
        elif(mode == "test"):
            self.datasets = self.datasets["test_matched" if data_args.task_name == "mnli" else "test"]
            self.sampler = SequentialSampler(self.datasets)
        # Log a few random samples from the training set:
        for index in random.sample(range(len(self.datasets)), 1):
            logger.info(f"Sample {index} of the training set: {self.datasets[index]}.")
           
        
        self.dataloader = DataLoader(
                        self.datasets, 
                        batch_size=batch_size, 
                        sampler=self.sampler, 
                        collate_fn=default_data_collator,
                        num_workers=4)
        self.iterator = enumerate(self.dataloader)
    
    def entity(self):
        return self.iterator
    
    def next(self):
        try:
            _, data = next(self.iterator)
        except Exception:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)  
        return data
        #return data[0], data[1]

    def __len__(self):
        return len(self.datasets)

class ComputeMetrics():
    def __init__(self, dataArgs):
        self.task_name = dataArgs.task_name
        
    def simple_accuracy(self, preds, labels):
        return (preds == labels).mean()
        
    def acc_and_f1(self, preds, labels):
        acc = self.simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }
        
    def pearson_and_spearman(self, preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }
        
            
    def result(self, labels, preds):
        if self.task_name == "cola":
            return {"mcc": matthews_corrcoef(labels, preds)}
        elif self.task_name == "sst2":
            return {"acc": self.simple_accuracy(preds, labels)}
        elif self.task_name == "mrpc":
            return self.acc_and_f1(preds, labels)
        elif self.task_name == "sts-b":
            return self.pearson_and_spearman(preds, labels)
        elif self.task_name == "qqp":
            return self.acc_and_f1(preds, labels)
        elif self.task_name == "mnli":
            return {"mnli/acc": self.simple_accuracy(preds, labels)}
        elif self.task_name == "mnli-mm":
            return {"mnli-mm/acc": self.simple_accuracy(preds, labels)}
        elif self.task_name == "qnli":
            return {"acc": self.simple_accuracy(preds, labels)}
        elif self.task_name == "rte":
            return {"acc": self.simple_accuracy(preds, labels)}
        elif self.task_name == "wnli":
            return {"acc": self.simple_accuracy(preds, labels)}
        elif self.task_name == "hans":
            return {"acc": self.simple_accuracy(preds, labels)}
        else:
            raise KeyError(self.task_name)    
     
     
     
     
     
     
     
     
     
     
     
     
     
     