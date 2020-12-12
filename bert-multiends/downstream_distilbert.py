from torch import nn
import torch
from transformers.configuration_bert import BertConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_bert import BertPreTrainedModel
from torch.nn import CrossEntropyLoss, MSELoss

class SequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        
        self.init_weights()
        
    def forward(self, input=None, labels=None, return_dict=True):
    
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        hidden_states = input[0]
        pooled_output = hidden_states[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)
        
        logits = self.classifier(pooled_output)
        loss = None
        
        if labels is not None:
            if self.num_labels == 1:
                # We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss() 
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                
        if not return_dict:
            output = (logits,) + input[2:]
            return ((loss,) + output) if loss is not None else output
    
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=input.hidden_states,
            attentions=input.attentions,
        )

    # def init_weights(self, module):
    
        # if isinstance(module, (nn.Linear, nn.Embedding)):
            # module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        # elif isinstance(module, nn.LayerNorm):
            # module.bias.data.zero_()
            # module.weight.data.fill_(1.0)
        # if isinstance(module, nn.Linear) and module.bias is not None:
            # module.bias.data.zero_()
        