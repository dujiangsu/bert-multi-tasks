import os
import sys
import logging
import torch
import time
import numpy as np
from torch import nn
import argparse
from downstream_add import SequenceClassification
from data import GlueDataArgs, GlueDataSets, ComputeMetrics, GlueIterator
# from transformers import BertConfig, BertTokenizer, BertModel
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertModel
from transformers import (
    # glue_compute_metrics,
    glue_tasks_num_labels
    )
       

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--learning-rate', type=float, default=0.00001)
# parser.add_argument('--batch-size', type=int, default=32)
args = parser.parse_args()
    
logger = logging.getLogger(__name__)

# Hyperparameters
# Define what tasks to train
tasks = ["SST-2", "MNLI", "STS-B", "QNLI"]
# Train 67k 393k 7k 108k   [67349, 392702, 5749, 104743]
# dev   872 20k  1.5k  5.7k
epochs = 6
model_name="sst2-nmli-sstb-qnli-distilbert-mtb-lr2"
# batch_size_train = [11, 64, 2, 19]
batch_size_train = [22, 128, 4, 38]
batch_size_val = [1, 1, 1, 1]
#batch_size = [44, 256, 5, 71]
bs = 128
learning_rate_0 = args.learning_rate
learning_rate_1 = 0.2
eval_interval = 1000
# weight_decay
frozen = 6000 # set 0 to prevent frozen the main model
bert_path="/home/nsccgz_jiangsu/bert-models/distilbert-base-cased"
cache_dir = os.path.join("/home/nsccgz_jiangsu/djs/output", model_name, "cache")
model_save_dir = os.path.join("/home/nsccgz_jiangsu/djs/output", model_name,"saved_model")

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
    datasets = list()
    # train_iter = list()
    # dev_iter = list()
    # test_iter = list()
    sub_optimizer = list()
    metrics = list()
    tokenizer = DistilBertTokenizer.from_pretrained(bert_path, cache_dir=cache_dir)
    
    for i in range(ntasks):    
        logger.info("Tasks:" + tasks[i])
        data_args.append(GlueDataArgs(task_name=tasks[i]))
        configuration.append(DistilBertConfig.from_pretrained(bert_path, num_labels=glue_tasks_num_labels[tasks[i].lower()], 
                                finetuning_task=data_args[i].task_name, cache_dir = cache_dir))
        if use_gpu:
            sub_models.append(SequenceClassification(configuration[i]).cuda())
        else: 
            sub_models.append(SequenceClassification(configuration[i]))
        
        datasets.append(GlueDataSets(data_args[i], tokenizer=tokenizer, cache_dir=cache_dir))
        sub_optimizer.append(torch.optim.AdamW(sub_models[i].parameters(), lr=learning_rate_0))
        metrics.append(ComputeMetrics(data_args[i]))
        logger.info("*** DataSet Ready ***")
    
    if use_gpu:
        Bert_model = DistilBertModel.from_pretrained(bert_path, return_dict=True).cuda()
    else:
        Bert_model = DistilBertModel.from_pretrained(bert_path, return_dict=True)
    
    bert_optimizer = torch.optim.Adamax(Bert_model.parameters(), lr=learning_rate_0)
    
    
    # balaned dataset
    train_num = list()    
    for i in range(ntasks):
        train_num.append(datasets[i].length("train"))
    #train_nummax = 
    #train_num = [x/train_nummax for x in train_num]
    print(train_num)
    iterations = (epochs * max(train_num) // bs) + 1
    #print(iterations)
    
    sub_scheduler = list()
    for i in range(ntasks):
        sub_scheduler.append(torch.optim.lr_scheduler.LambdaLR(sub_optimizer[i], lambda step: 1 if step<=frozen else (1.0-step/iterations))) #if step <= frozen else learning_rate_1)    
    Bert_scheduler = torch.optim.lr_scheduler.LambdaLR(bert_optimizer, lambda step: 1 if step<=frozen else (1.0-step/iterations))# if step <= frozen else learning_rate_1
    
    # datasets[i].dataloader("train", batch_size_train[i])
    train_iter = list()
    for i in range(ntasks):
        train_iter.append(GlueIterator(datasets[i].dataloader("train", batch_size_train[i])))
    
    for i in range(1, iterations+1):
                
        if i > frozen:
            for p in Bert_model.parameters():
                p.requires_grad = True
            Bert_model.train()
        elif i == frozen:
            for p in Bert_model.parameters():
                p.requires_grad = True
            Bert_model.train()   
            logging.info("#####################################")
            logging.info("Release the Traing of the Main Model.")
            logging.info("#####################################")
        else:
            for p in Bert_model.parameters():
                p.requires_grad = False
            Bert_model.eval()
        
        losses=list()
        loss_rates=list()
        
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
   
        
        losssum = sum(losses).item()     
        for j in range(ntasks):
            loss_rates.append(4*losses[j].item()/losssum)
        
        loss = 0
        printInfo = 'TOTAL/Train {}/{}, lr:{}'.format(i, iterations, Bert_scheduler.get_lr())
        for j in range(ntasks):
            loss += losses[j] * batch_size_train[j] * loss_rates[j]
            printInfo += ', loss{}-{:.6f}'.format(j,losses[j])
            sub_optimizer[j].zero_grad()
            
        logging.info(printInfo) 
        
        if i > frozen:
            bert_optimizer.zero_grad()
        loss.backward()
        
        if i > frozen:
            bert_optimizer.step()
            
        for j in range(ntasks):
            sub_optimizer[j].step()
            sub_scheduler[j].step()
        
        Bert_scheduler.step()
        
        if (i % eval_interval == 0):
            evaluate(Bert_model, sub_models, datasets, batch_size_val, metrics, ntasks)
            # save_models(Bert_model, sub_models, ntasks, i)
    
    evaluate(Bert_model, sub_models, datasets, batch_size_val, metrics, ntasks)
    # save_models(Bert_model, sub_models, ntasks, iterations)

    
    
def save_models(main_model, sub_models, ntasks, iterations):
    for i in range(ntasks):
        sub_models[i].save_pretrained(os.path.join(model_save_dir, "{}-checkpoint-{:06}.pth.tar".format(tasks[i], iterations)))
    main_model.save_pretrained(os.path.join(model_save_dir, "{}-checkpoint-{:06}.pth.tar".format("main", iterations)))

def evaluate(main_model, sub_models, datasets, batch_sizes, metrics, ntasks):   
 
    for i in range(ntasks):
        if datasets[i].task_name == "mnli": 
            evaluate_each(main_model, sub_models[i], datasets[i], batch_sizes[i], metrics[i], mm="validation_mismatched")
        
        evaluate_each(main_model, sub_models[i], datasets[i], batch_sizes[i], metrics[i])
        
    

def evaluate_each(main_model, sub_model, dataset, bs, metrics, mm="validation_matched"):

    all_labels = []
    all_preds = []
    all_losses = []
    
    iterations = dataset.length("dev", mm) // bs
    eval_iter = GlueIterator(dataset.dataloader("dev", bs, mm))
    
    printInfo = "*** Evaluation of {:s} ***".format(metrics.task_name)
    logging.info(printInfo)

    with torch.no_grad():
        for i in range(1, iterations+1):
        
            main_model.eval()
            sub_model.eval()
            data = eval_iter.next()
        
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
            
            if metrics.num_labels == 3:                
                pred = [x.index(max(x)) for x in output.logits.cpu().numpy().tolist()]
            elif metrics.num_labels == 2:
                pred = np.round(softmax_layer(output.logits).cpu().t()[1].numpy()).tolist()
            elif metrics.num_labels == 1:
                #print(np.array(label).shape)
                pred = output.logits.cpu().t().numpy().tolist()[0]
                
            # pred = np.round(softmax_layer(output.logits).cpu().t()[1].numpy()).tolist()
            
            
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
