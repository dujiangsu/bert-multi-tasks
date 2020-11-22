import os
import sys
import logging
import torch
import time
from torch import nn
from downstream import SequenceClassification
from data import GlueDataArgs, DataIterator
from transformers import BertConfig, BertTokenizer, BertModel
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)
        
class GlueTraingArgs:
    def __init__(self,output_dir="/GPUFS/nsccgz_xliao_djs/tqr/result",
    do_train=False,do_eval=False,do_predict=False):
        self.output_dir=output_dir
        self.do_train=do_train
        self.do_eval=do_eval
        self.do_predict=do_predict
    
logger = logging.getLogger(__name__)

epochs = 3
batch_size = 32
bert_path="/GPUFS/nsccgz_xliao_djs/bert_multiend/bert-model/bert-base-cased"
task0 = "CoLA"
task1 = "SST-2"
data_task0 = "/GPUFS/nsccgz_xliao_djs/glue_dataset/cola"
data_task1 = "/GPUFS/nsccgz_xliao_djs/glue_dataset/sst2"
cache_dir = "/GPUFS/nsccgz_xliao_djs/tqr/bert-cache/"

use_gpu = torch.cuda.is_available()

# TODO: GPU Training.
def main():
    training_args = GlueTraingArgs(do_train=True)
    data_args_task0 = GlueDataArgs(task_name = task0, data_dir = data_task0)
    data_args_task1 = GlueDataArgs(task_name = task1, data_dir = data_task1)
    
    if use_gpu:
        print("Training on GPU.")
    
    # logging
    log_format = '[%(asctime)s] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%d %I:%M:%S')
    t = time.time()
    local_time = time.localtime(t)
    if not os.path.exists('./log'):
        os.mkdir('./log')
    fh = logging.FileHandler(os.path.join('log/train-{}{:02}{}'.format(local_time.tm_year % 2000, local_time.tm_mon, t)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    
    config_task0 = BertConfig.from_pretrained(
        bert_path,
        num_labels=glue_tasks_num_labels[data_args_task0.task_name], 
        finetuning_task=data_args_task0.task_name,
        cache_dir=cache_dir
    )
    
    config_task1 = BertConfig.from_pretrained(
        bert_path,
        num_labels=glue_tasks_num_labels[data_args_task1.task_name], 
        finetuning_task=data_args_task1.task_name,
        cache_dir=cache_dir
    )
    # Model Prepare, The Bert Model has loaded the pretrained model, 
    # and these downstream structures are initialized randomly.
    # TODO: Adding Seed for random.  referee: Trainer.train()
    
    if use_gpu:
        model_Bert = BertModel.from_pretrained(bert_path, return_dict=True).cuda()
        model_task0 = SequenceClassification(config_task0).cuda()
        model_task1 = SequenceClassification(config_task1).cuda()
    else:
        model_Bert = BertModel.from_pretrained(bert_path, return_dict=True)
        model_task0 = SequenceClassification(config_task0)
        model_task1 = SequenceClassification(config_task1)
    
    # Data prepare
    tokenizer = BertTokenizer.from_pretrained(bert_path, cache_dir=cache_dir)    
    data_iterator_train_task0 = DataIterator(data_args_task0, tokenizer=tokenizer, mode="train", cache_dir=cache_dir, batch_size=batch_size)
    # data_iterator_eval_task0 = DataIterator(data_args_task0, tokenizer=tokenizer, mode="dev", cache_dir=cache_dir_task0, batch_size=batch_size) 
    data_iterator_train_task1 = DataIterator(data_args_task1, tokenizer=tokenizer, mode="train", cache_dir=cache_dir, batch_size=batch_size)
    # data_iterator_eval_task1 = DataIterator(data_args_task1, tokenizer=tokenizer, mode="dev", cache_dir=cache_dir_task1,batch_size=batch_size)
    
    logger.info("*** DataSet Ready ***")
    
    opt_bert = torch.optim.AdamW(model_Bert.parameters(), lr=0.0001)
    opt_task0 = torch.optim.AdamW(model_task0.parameters(), lr=0.0001) 
    opt_task1 = torch.optim.AdamW(model_task1.parameters(), lr=0.0001)
    
    iterations = (epochs * len(data_iterator_train_task1) // batch_size) + 1  
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt_bert, lambda step: (1.0-step/iterations))
    print(iterations)

    all_iters = 0


    for i in range(1, iterations+1):
    
        all_iters += 1        
        scheduler.step()
        
        model_Bert.train()
        model_task0.train()
        model_task1.train()
        
        data0 = data_iterator_train_task0.next()
        data1 = data_iterator_train_task1.next()
        
        if use_gpu:        
            input_ids0=data0['input_ids'].cuda()
            attention_mask0=data0['attention_mask'].cuda()
            token_type_ids0=data0['token_type_ids'].cuda()
            input_ids1=data1['input_ids'].cuda()
            attention_mask1=data1['attention_mask'].cuda()
            token_type_ids1=data1['token_type_ids'].cuda()
            label0=data0['labels'].cuda()
            label1=data1['labels'].cuda()
        else:
            input_ids0=data0['input_ids']
            attention_mask0=data0['attention_mask']
            token_type_ids0=data0['token_type_ids']
            input_ids1=data1['input_ids']
            attention_mask1=data1['attention_mask']
            token_type_ids1=data1['token_type_ids']     
            label0=data0['labels']
            label1=data1['labels']
        
        output_inter0 = model_Bert(input_ids=input_ids0, attention_mask=attention_mask0, token_type_ids=token_type_ids0, return_dict=True)
        output_inter1 = model_Bert(input_ids=input_ids1, attention_mask=attention_mask1, token_type_ids=token_type_ids1, return_dict=True)
        
        loss0 = model_task0(input=output_inter0, labels=label0)[0]
        loss1 = model_task1(input=output_inter1, labels=label1)[0]

        ratio = loss0/loss1
        weight0 = (2*ratio) / (1+ratio) # solution of equations: weight0/weight1 == loss0/loss1 & weight0+weight1 == 2
        weight1 = 2 - weight0
        
        loss = loss0*weight0 + loss1*weight1
        
        # printInfo = 'TRAIN ITER {}: loss ={:.6f}, loss0={:.6f}, loss1={:.6f}'.format(all_iters, loss, loss0, loss1)
        printInfo = 'TOTAL/Train {}/{}:lr:{}, loss={:.6f}, loss0={:.6f}, loss1={:.6f}'.format(all_iters, iterations, scheduler.get_lr(), loss, loss0, loss1)
        logging.info(printInfo)
        
        
        opt_bert.zero_grad()
        opt_task0.zero_grad()
        opt_task1.zero_grad()
        # loss0.backward()
        loss.backward()
        
        opt_bert.step()
        opt_task0.step()
        opt_task1.step()

'''
def save(model):
    model.save_model(path)
    model0.save_model(path)
    model1.save_model(path)
'''

if __name__ == "__main__":
    main()

