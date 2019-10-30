import torch
import torch.nn as nn
from transformers import BertModel

# At this point, I'm using BERT as a blackbox model, so I'm adapting this class from here:
# https://github.com/kabirahuja2431/FineTuneBERT/blob/master/src/model.py

class BERTSentimentClassifier(nn.Module):

    def __init__(self, freeze_bert = True):
        super(BERTSentiment, self).__init__()
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')

        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        
        self.cls_layer = nn.Linear(768, 1)

    def forward(self, seq, attention_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attention_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''
        cont_reps, _ = self.bert_layer(seq, attention_mask=attention_masks)
        cls_rep = cont_reps[:, 0]
        logits = self.cls_layer(cls_rep)
        return logits
