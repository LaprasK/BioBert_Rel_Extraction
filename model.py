from torch import nn
from pytorch_pretrained_bert import BertModel



class Rel_Net(nn.Module):
    
    def __init__(self, config, bert_state_dict = None, number_class = 8):
        super(Rel_Net, self).__init__()
        self.number_class = number_class
        self.bert = BertModel(config)
        if bert_state_dict:
            self.bert.load_state_dict(bert_state_dict)
        self.bert.eval()
        self.decoder = nn.LSTM(input_size=768, hidden_size=768//2, num_layers = 2, \
                               batch_first= True, bidirectional = True )
        self.dropout = nn.Dropout(config.hidden_dropout_prob) 
        self.linear = nn.Linear(config.hidden_size, self.number_class)
    
    
    
    
    def forward(self, input_ids, token_type_ids = None, input_mask=None, labels= None, rel_labels = None):
        output, _ = self.bert(input_ids, token_type_ids, input_mask, output_all_encoded_layers = False)
        output = self.dropout(output)
        output, _ = self.decoder(output)
        output = self.dropout(output)
        logits = self.linear(output)
        
        if labels is not None and rel_labels = None:
            criterion = nn.CrossEntropyLoss()
            
            if input_mask is not None:
                output_mask = input_mask.view(-1) == 1
                logits_mask = logits.view(-1, self.number_class)[output_mask]
                label_mask = labels.view(-1)[output_mask]
                loss = criterion(logits_mask, label_mask)
                
            else:
                loss = criterion(logits.view(-1, self.number_class), labels.view(-1))
            
            return loss
        else:
            return logits