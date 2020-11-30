# The pre-training model, e.g. Bert, trains a substantial model as basis. 
# The fundamental model contains most knowledge of natural language.
# Then, based on these knowledge, new structures are added and fine-tuned for
# different kinds of tasks.
# Because most knowledge is stored in the fundamental structure rather than these extra 
# structures, we try to fine-tune multiple downstream tasks together.
# Target: A fundamental pre-training model can support multiple downstream tasks.
# Fundamental model: Bert.
# Downstream tasks (GLUE): 
# CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI.
# ulimit -n 2000

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
    #glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)



        
# class GlueTraingArgs:
    # def __init__(self,output_dir="/home/dujiangsu/output/results",
    # do_train=False,do_eval=False,do_predict=False):
        # self.output_dir=output_dir
        # self.do_train=do_train
        # self.do_eval=do_eval
        # self.do_predict=do_predict
    
logger = logging.getLogger(__name__)


epochs = 20
batch_size = 32
bert_path="/home/dujiangsu/bert-base-cased"
task0 = "SST-2"
task1 = "QQP"
data_task0 = "/home/dujiangsu/glue_dataset/SST-2"
data_task1 = "/home/dujiangsu/glue_dataset/QQP"
cache_dir = "/home/dujiangsu/output/"

use_gpu=torch.cuda.is_available()

# TODO: GPU Training.
def main():
    #training_args = GlueTraingArgs(do_train=True)
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
    
    print(config_task1)
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
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt_bert,
                                                lambda step: (1.0-step/iterations))
    print(iterations)
    
    
    data0 = data_iterator_train_task0.next()
    
    print(data0)
    
    input_ids0=data0['input_ids']
    attention_mask0=data0['attention_mask']
    token_type_ids0=data0['token_type_ids']  
    label0=data0['labels']
    
    print(input_ids0)
    print(input_ids0.size())
    print(input_ids0.type())
    print(attention_mask0)
    print(attention_mask0.size())
    print(attention_mask0.type())
    print(token_type_ids0)
    print(token_type_ids0.size())
    print(token_type_ids0.type())
    print(label0)
    print(label0.size())
    print(label0.type())
    
    
    all_iters = 0
  
'''    
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
        
        loss = loss0+loss1
        
        # printInfo = 'TRAIN ITER {}: loss ={:.6f}, loss0={:.6f}, loss1={:.6f}'.format(all_iters, loss, loss0, loss1)
        printInfo = 'TOTAL/Train {}/{}:lr:{} , loss0={:.6f}, loss0={:.6f}, loss1={:.6f}'.format(all_iters, iterations, scheduler.get_lr(), loss, loss0, loss1)
        logging.info(printInfo)
        
        # print(loss)
        # print(all_iters)
        
        opt_bert.zero_grad()
        opt_task0.zero_grad()
        opt_task1.zero_grad()
        # loss0.backward()
        loss.backward()
        
        opt_bert.step()
        opt_task0.step()
        opt_task1.step()


'''

if __name__ == "__main__":
    main()