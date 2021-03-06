import torch
import time
import numpy as np
from transformers import BertConfig, BertTokenizer, BertModel




        
def main():

# Env
    use_gpu=False
    #torch.cuda.is_available()
    
    # batchsize = [1,4,8,16,32,64,128,256]

# Data Augment
    input_np = np.random.randint(1000, size=(1,512))
    
    input_ids=torch.from_numpy(input_np).long()
    attention_mask=torch.zeros(1, 512).long()
    token_type_ids=torch.ones(1, 512).long()
    
    if use_gpu:
        input_ids=input_ids.cuda()
        attention_mask=attention_mask.cuda()
        token_type_ids=token_type_ids.cuda()
        

    

# Model Prepare

    configuration = BertConfig(vocab_size=30522, hidden_size=768, num_hudden_layers=12,
                            num_attention_heads=12, intermediate_size=3072, 
                            hidden_act='gelu', hidden_dropout_prob=0.1, 
                            attention_probs_dropout_prob=0.1, 
                            max_position_embeddings=512, type_vocab_size=2, 
                            initializer_range=0.02, layer_norm_eps=1e-12, 
                            pad_token_id=0, gradient_checkpointing=False)
    
    if use_gpu:
        model = BertModel(configuration).cuda()
    else:
        model = BertModel(configuration)
    
# Eval with Speed Record
    
    model.eval()
    
    t1 = time.time()
    with torch.no_grad():
        for i in range(512):
            print(i)
            output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    print('val_time = {:.6f}'.format(time.time() - t1))
    





if __name__ == "__main__":
    main()
