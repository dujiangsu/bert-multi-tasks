import torch
import time
from transformers import BertConfig, BertTokenizer, BertModel


class GlueTraingArgs:
    def __init__(self,output_dir="/home/dujiangsu/output/results",
    do_train=False,do_eval=False,do_predict=False):
        self.output_dir=output_dir
        self.do_train=do_train
        self.do_eval=do_eval
        self.do_predict=do_predict


        
def main():

# Env
    use_gpu=torch.cuda.is_available()
    
    
# Data Augment

# Model Prepare

    configuration = BertConfig()
    
    if use_gpu:
        model = BertModel(configuration).cuda()

# Eval with Speed Record







if __name__ == "__main__":
    main()