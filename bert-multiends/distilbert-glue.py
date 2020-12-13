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
import numpy as np
from torch import nn
from downstream_distilbert import SequenceClassification
from data import GlueDataArgs, DataIterator, ComputeMetrics
# from transformers import BertConfig, BertTokenizer, BertModel
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertModel
from transformers import (
    # glue_compute_metrics,
    glue_tasks_num_labels
    )
       
    
logger = logging.getLogger(__name__)

# Hyperparameters
epochs = 20
model_name="distilbert-base-cased"
batch_size = [24, 128, 4, 2]
bs = 128
batch_size_val = [109, 200, 200, 102]
learning_rate = 0.00001
eval_interval = 10
bert_path="/home/nsccgz_jiangsu/bert-models/distilbert-base-cased"
cache_dir = os.path.join("/home/nsccgz_jiangsu/djs/output", model_name, "cache")
model_save_dir = os.path.join("/home/nsccgz_jiangsu/djs/output", model_name,"saved_model")


# Define what tasks to train
tasks = ["SST-2", "QQP", "CoLA", "MRPC"]



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

use_gpu=torch.cuda.is_available()
if use_gpu:
    print("Training on GPU.")

def main():
    
    
    
    ntasks = len(tasks)
    
    data_args = list()
    configuration = list()
    sub_models = list()
    train_iter = list()
    dev_iter = list()
    test_iter = list()
    sub_optimizer = list()
    metrics = list()
    tokenizer = DistilBertTokenizer.from_pretrained(bert_path, cache_dir=cache_dir)
    
    for i in range(ntasks):    
        logger.info("Tasks:" + tasks[i])
        data_args.append(GlueDataArgs(task_name=tasks[i]))
        configuration.append(DistilBertConfig.from_pretrained(bert_path, num_labels=glue_tasks_num_labels[data_args[i].task_name], 
                                finetuning_task=data_args[i].task_name, cache_dir = cache_dir))
        if use_gpu:
            sub_models.append(SequenceClassification(configuration[i]).cuda())
        else: 
            sub_models.append(SequenceClassification(configuration[i]))
            
        train_iter.append(DataIterator(data_args[i], tokenizer=tokenizer, mode="train", cache_dir=cache_dir, batch_size=batch_size[i]))
        dev_iter.append(DataIterator(data_args[i], tokenizer=tokenizer, mode="dev", cache_dir=cache_dir, batch_size=batch_size_val[i]))
        
        sub_optimizer.append(torch.optim.AdamW(sub_models[i].parameters(), lr=learning_rate))
        
        metrics.append(ComputeMetrics(data_args[i]))
        
        logger.info("*** DataSet Ready ***")
    
    if use_gpu:
        Bert_model = DistilBertModel.from_pretrained(bert_path, return_dict=True).cuda()
    else:
        Bert_model = DistilBertModel.from_pretrained(bert_path, return_dict=True)
    
    bert_optimizer = torch.optim.AdamW(Bert_model.parameters(), lr=learning_rate)
    
    
    # balaned dataset
    train_num = list()    
    for i in range(ntasks):
        train_num.append(len(train_iter[i]))
    #train_nummax = 
    #train_num = [x/train_nummax for x in train_num]
    #print(train_num)
    iterations = (epochs * max(train_num) // bs) + 1
    #print(iterations)
    
    sub_scheduler = list()
    for i in range(ntasks):
        sub_scheduler.append(torch.optim.lr_scheduler.LambdaLR(sub_optimizer[i], lambda step: (1.0-step/iterations)))    
    Bert_scheduler = torch.optim.lr_scheduler.LambdaLR(bert_optimizer, lambda step: (1.0-step/iterations))
    
    for i in range(1, iterations+1):
        Bert_model.train()
        losses=list()
        for j in range(ntasks):
            sub_models[j].train()
            data = train_iter[j].next()
            
            if use_gpu:
                input_ids=data['input_ids'].cuda()
                attention_mask=data['attention_mask'].cuda()
                #token_type_ids=data['token_type_ids'].cuda()
                label=data['labels'].cuda()
            else:
                input_ids=data['input_ids']
                attention_mask=data['attention_mask']
                #token_type_ids=data['token_type_ids']
                label=data['labels']
                
            output_inter = Bert_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True) # token_type_ids=token_type_ids,
            losses.append(sub_models[j](input=output_inter, labels=label)[0])
            
        loss = 0
        printInfo = 'TOTAL/Train {}/{}, lr:{}'.format(i, iterations, Bert_scheduler.get_lr())
        for j in range(ntasks):
            loss += losses[j] * batch_size[j]
            printInfo += ', loss{}-{:.6f}'.format(j,losses[j])
            sub_optimizer[j].zero_grad()
            
        logging.info(printInfo)   
        bert_optimizer.zero_grad()
        loss.backward()
        
        bert_optimizer.step()
        for j in range(ntasks):
            sub_optimizer[j].step()
            sub_scheduler[j].step()
        Bert_scheduler.step()
        
        if (i % eval_interval == 0):
            for j in range(ntasks):
                evaluate(Bert_model, sub_models[j], dev_iter[j], batch_size_val[j], metrics[j])
                sub_models[j].save_pretrained(os.path.join(model_save_dir, "{}-checkpoint-{:06}.pth.tar".format(tasks[j], i)))
            Bert_model.save_pretrained(os.path.join(model_save_dir, "{}-checkpoint-{:06}.pth.tar".format("main", i)))
    
    
    for i in range(ntasks):
        evaluate(Bert_model, sub_models[i], dev_iter[i], batch_size_val[i], metrics[i])
        sub_models[i].save_pretrained(os.path.join(model_save_dir, "{}-checkpoint-{:06}.pth.tar".format(tasks[j], iterations)))
            
    Bert_model.save_pretrained(os.path.join(model_save_dir, "{}-checkpoint-{:06}.pth.tar".format("main", iterations)))    
    
    

def evaluate(main_model, sub_model, dataset, bs, metrics):

    all_labels = []
    all_preds = []
    all_losses = []
    
    iterations = len(dataset) // bs
    
    printInfo = "*** Evaluation of {:s} ***".format(metrics.task_name)
    logging.info(printInfo)

    with torch.no_grad():
        for i in range(1, iterations+1):
        
            main_model.eval()
            sub_model.eval()
            data = dataset.next()
        
            if use_gpu:        
                input_ids = data['input_ids'].cuda()
                attention_mask = data['attention_mask'].cuda()
                # token_type_ids = data['token_type_ids'].cuda()
                label = data['labels'].cuda()
            else:
                input_ids = data['input_ids']
                attention_mask = data['attention_mask']
                # token_type_ids = data['token_type_ids']
                label = data['labels']
            
            output_inter = main_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            output = sub_model(input=output_inter, labels=label)
            loss = output[0].cpu().numpy().tolist()
            label = label.cpu().numpy().tolist()
            softmax_layer = torch.nn.Softmax(dim=1)
            pred = np.round(softmax_layer(output.logits).cpu().t()[1].numpy()).tolist()
            
            
            all_labels += label
            all_preds += pred
            all_losses += [loss]
            
    
    logging.info("loss = {:.6f}".format(sum(all_losses)/len(all_losses)))
    
    eval_result = metrics.result(np.array(all_labels), np.array(all_preds))
    
    for i in eval_result:
        printInfo = "{:s} = {:.6f}".format(i, eval_result[i])
        logging.info(printInfo)    
    
    

if __name__ == "__main__":
    main()
