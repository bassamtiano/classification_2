import sys
import torch
import torch.nn as nn

from transformers import BertModel

class MultiClassModel(nn.Module):
    def __init__(self,
                 n_out,
                 dropout) -> None:
        super(MultiClassModel, self).__init__()
        
        # Inisiaisasi Language Model
        self.bert = BertModel.from_pretrained('indolem/indobert-base-uncased', output_hidden_states = True)
        
        # Disimpan di memori lokal / class sendiri
        self.pre_classifier = nn.Linear(768, 768)
        
        # Stabilisasi -1 -> 1
        self.activation = nn.Tanh()
        
        # Mencegah monoton
        self.dropout = nn.Dropout(dropout)
        
        # Merubah hasil output ke target jumlah label
        self.classifier = nn.Linear(768, n_out)
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        # Mengambil output dari LM
        bert_out = self.bert(input_ids = input_ids, 
                             attention_mask = attention_mask,
                             token_type_ids = token_type_ids)[0]
        pooler = bert_out[:, 0]
        
        sys.exit()
        # Menyimpan di memori class lokal
        # pooler = self.pre_classifier(pooler)
        # # Stabilisasi
        # pooler = self.activation(pooler)
        # # Mecegah monoton
        # output = self.dropout(pooler)
        # # Merubah hasil output
        # output = self.classifier(output)
        
        # return output
    
