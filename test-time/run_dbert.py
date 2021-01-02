import torch
import time
import numpy as np
from transformers import DistilBertConfig, BertTokenizer
from transformers.modeling_distilbert import Embeddings
from dbert_without_embed import DistilBertModel

def main():

# Env
    # use_gpu=False
    use_gpu = torch.cuda.is_available()
    print(use_gpu)
    
# Data Augment
    input_np = np.random.randint(1000, size=(1, 128))
    
    input_ids=torch.from_numpy(input_np).long()
    attention_mask=torch.zeros(1, 128).long()
    token_type_ids=torch.ones(1, 128).long()
    
    if use_gpu:
        input_ids=input_ids.cuda()
        attention_mask=attention_mask.cuda()
        token_type_ids=token_type_ids.cuda()

# Embeddings
    configuration = DistilBertConfig(vocab_size=30522, hidden_dim=3072,
                            n_layers=6, n_heads=12,
                            hidden_act='gelu', hidden_dropout_prob=0.1, 
                            attention_probs_dropout_prob=0.1, 
                            max_position_embeddings=128, type_vocab_size=2, 
                            initializer_range=0.02, layer_norm_eps=1e-12, 
                            pad_token_id=0, gradient_checkpointing=False)

    t0 = time.time()

    # if use_gpu:
    #     embed = Embeddings(configuration).cuda()
    # else:
    embed = Embeddings(configuration)

    t1 = time.time()

# Model Prepare
    if use_gpu:
        model = DistilBertModel(configuration, embed).cuda()
    else:
        model = DistilBertModel(configuration, embed)
        
    # features = model.state_dict().keys()
    # print(features)
    
# Eval with Speed Record    
    t2 = time.time()
    model.eval()
    
    with torch.no_grad():
        for i in range(100):
            output = model(input_ids=input_ids)
    
    t3 = time.time()

    print('embedding = {:.6f} | load_model = {:.6f} | val_time = {:.6f}'.format(t1-t0, t2-t1, t3-t2))
    

if __name__ == "__main__":
    main()
