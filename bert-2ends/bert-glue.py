import os
import sys
import logging
import torch
import time
import numpy as np
from torch import nn
from downstream import SequenceClassification
from data import GlueDataArgs, DataIterator, ComputeMetrics
from transformers import BertConfig, BertTokenizer, BertModel
from transformers import (
    glue_tasks_num_labels,
    glue_compute_metrics
)
        
class GlueTrainArgs:
    def __init__(self, output_dir="/GPUFS/nsccgz_xliao_djs/bert-multi-tasks/result",
    do_train=False, do_eval=False, do_predict=False):
        self.output_dir = output_dir
        self.do_train = do_train
        self.do_eval = do_eval
        self.do_predict = do_predict
    
logger = logging.getLogger(__name__)

epochs = 10
batch_size = 32
learning_rate = 0.0001
eval_interval = 10
bert_path = "/home/dujiangsu/bert-base-cased"
task0 = "CoLA"
task1 = "SST-2"
data_task0 = "/home/dujiangsu/tqr/glue_dataset/cola"
data_task1 = "/home/dujiangsu/tqr/glue_dataset/sst2"
cache_dir = "/home/dujiangsu/tqr/cache/"
model_save_dir = "/home/dujiangsu/tqr/bert-multi-tasks/saved_model/"

use_gpu = torch.cuda.is_available()

def main():
    training_args = GlueTrainArgs(do_train=True)
    data_args_task0 = GlueDataArgs(task_name=task0, data_dir=data_task0)
    data_args_task1 = GlueDataArgs(task_name=task1, data_dir=data_task1)

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
        num_labels = glue_tasks_num_labels[data_args_task0.task_name], 
        finetuning_task = data_args_task0.task_name,
        cache_dir = cache_dir
    )
    
    config_task1 = BertConfig.from_pretrained(
        bert_path,
        num_labels = glue_tasks_num_labels[data_args_task1.task_name], 
        finetuning_task = data_args_task1.task_name,
        cache_dir = cache_dir
    )
    
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
    data_iterator_train_task1 = DataIterator(data_args_task1, tokenizer=tokenizer, mode="train", cache_dir=cache_dir, batch_size=batch_size)
    data_iterator_eval_task0 = DataIterator(data_args_task1, tokenizer=tokenizer, mode="dev", cache_dir=cache_dir, batch_size=batch_size)
    data_iterator_eval_task1 = DataIterator(data_args_task1, tokenizer=tokenizer, mode="dev", cache_dir=cache_dir, batch_size=batch_size)
    logger.info("*** Dataset Ready ***")

    data=data_iterator_train_task0.next()
    # print(data)
    # print(data.keys())
    
    opt_main = torch.optim.AdamW(model_Bert.parameters(), lr=learning_rate)
    opt_task0 = torch.optim.AdamW(model_task0.parameters(), lr=learning_rate) 
    opt_task1 = torch.optim.AdamW(model_task1.parameters(), lr=learning_rate)

    metrics_task0 = ComputeMetrics(data_args_task0)
    metrics_task1 = ComputeMetrics(data_args_task1)
    
    # iterations = (epochs * len(data_iterator_train_task1) // batch_size) + 1
    iterations = 200
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt_main, lambda step: (1.0-step/iterations))
    all_iters = 0

    for i in range(1, iterations+1):
        all_iters += 1        
        scheduler.step()
        model_Bert.train()
        model_task0.train()
        model_task1.train()
        data0 = data_iterator_train_task0.next()
        data1 = data_iterator_train_task1.next()
        # print()

        if use_gpu:        
            input_ids0 = data0['input_ids'].cuda()
            attention_mask0 = data0['attention_mask'].cuda()
            token_type_ids0 = data0['token_type_ids'].cuda()
            label0 = data0['labels'].cuda()
            input_ids1 = data1['input_ids'].cuda()
            attention_mask1 = data1['attention_mask'].cuda()
            token_type_ids1 = data1['token_type_ids'].cuda()
            label1 = data1['labels'].cuda()
        else:
            input_ids0 = data0['input_ids']
            attention_mask0 = data0['attention_mask']
            token_type_ids0 = data0['token_type_ids']
            label0 = data0['labels']
            input_ids1 = data1['input_ids']
            attention_mask1 = data1['attention_mask']
            token_type_ids1 = data1['token_type_ids']
            label1 = data1['labels']
        
        output_inter0 = model_Bert(input_ids=input_ids0, attention_mask=attention_mask0, token_type_ids=token_type_ids0, return_dict=True)
        output_inter1 = model_Bert(input_ids=input_ids1, attention_mask=attention_mask1, token_type_ids=token_type_ids1, return_dict=True)
        
        loss0 = model_task0(input=output_inter0, labels=label0)[0]
        loss1 = model_task1(input=output_inter1, labels=label1)[0]

        # balance the losses of sub-tasks
        ratio = loss0/loss1
        weight0 = (2*ratio) / (1+ratio) # solution of equations: weight0/weight1 == loss0/loss1 & weight0+weight1 == 2
        weight1 = 2 - weight0
        loss = loss0*weight0 + loss1*weight1

        printInfo = 'TOTAL/Train {}/{}: lr:{}, loss={:.6f}, loss0={:.6f}, loss1={:.6f}'.format(all_iters, iterations, scheduler.get_lr(), loss, loss0, loss1)
        logging.info(printInfo)
    
        opt_main.zero_grad()
        opt_task0.zero_grad()
        opt_task1.zero_grad()
        loss.backward()
        opt_main.step()
        opt_task0.step()
        opt_task1.step()

        if i % eval_interval == 0:
            evaluate(model_Bert, model_task0, data_iterator_eval_task0, metrics_task0)
            evaluate(model_Bert, model_task1, data_iterator_eval_task1, metrics_task1)

    if iterations % eval_interval != 0:
        evaluate(model_Bert, model_task0, data_iterator_eval_task0, metrics_task0)
        evaluate(model_Bert, model_task1, data_iterator_eval_task1, metrics_task1)

    # Saving models
    model_Bert.save_pretrained(model_save_dir + "main")
    model_task0.save_pretrained(model_save_dir + "task0")
    model_task1.save_pretrained(model_save_dir + "task1")


def evaluate(main_model, sub_model, dataset, metrics):

    all_labels = []
    all_preds = []
    # all_labels = np.empty(0, dtype=int)
    # all_preds = np.empty(0, dtype=int)
    
    printInfo = "*** Evaluation of {:s} ***".format(metrics.task_name)
    logging.info(printInfo)

    with torch.no_grad():
        for i in range(1, len(dataset)+1):

            main_model.eval()
            sub_model.eval()
            data = dataset.next()

            if use_gpu:        
                input_ids = data['input_ids'].cuda()
                attention_mask = data['attention_mask'].cuda()
                token_type_ids = data['token_type_ids'].cuda()
                label = data['labels'].cuda()
            else:
                input_ids = data['input_ids']
                attention_mask = data['attention_mask']
                token_type_ids = data['token_type_ids']
                label = data['labels']

            output_inter = main_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
            output = sub_model(input=output_inter, labels=label, return_dict=True)
            loss = output[0]
            label = label.cpu().numpy().tolist()
            softmax_layer = torch.nn.Softmax(dim=1)
            pred = np.round(softmax_layer(output.logits).cpu().t()[0].numpy()).tolist()

            all_labels += label
            all_preds += pred
            # np.append(all_labels, label)
            # np.append(all_preds, pred)

    eval_result = metrics.result(np.array(all_labels), np.array(all_preds))
    printInfo = "loss = {:.6f}".format(loss)
    logging.info(printInfo)
    for i in eval_result:
        printInfo = "{:s} = {:.6f}".format(i, eval_result[i])
        logging.info(printInfo)

'''
def load_model(training_args, config, path):
    path = path
'''

    
if __name__ == "__main__":
    main()

