import json
import copy
from pytorch_pretrained_bert import BertTokenizer
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split, Dataset
import torch

class Config(object):
    def __init__(self, config_json):
        """json should contain data_path, vocab_path, tags_vocab, batch_size, num_epochs, lr, bert_weight_path, bert_conf_path"""
        if isinstance(config_json, str):
            with open(config_json, 'r', encoding='utf-8') as f:
                json_config = json.loads(f.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        else:
            raise ValueError('Must be str path of config')
        self.label_dict = {"T-concept": 7}
        self.idx2tag = ['O', 'B-Test', 'B-Problem', 'B-Treat', 'I-Test', 'I-Problem', 'I-Treat', 'I-Piece']
  
    def __repr__(self):
        return str(self.to_json_string())
    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
    def save(self, path):
        with open(path, 'w') as f:
            f.write(self.to_json_string())

            
def pad_batch(batch, max_len=None, pad=0, idO=0):
    """pad a batch to max len of seq if max_len is None"""
    sents = [s[0].copy() for s in batch]
    sents_len = [len(s) for s in sents]
    labels = [s[1].copy() for s in batch]
    matrix = [s[-1] for s in batch]
    #atts = [s[-1] for s in batch]
    if max_len is None:
        max_len = max(sents_len)
    atts = [None] * len(sents)
    for i in range(len(sents)):
        atts[i] = [1] * sents_len[i] + [0] * (max_len - sents_len[i])
        sents[i].extend([pad] * (max_len - sents_len[i]))
        labels[i].extend([idO] * (max_len - sents_len[i]))
    return torch.LongTensor(sents), torch.LongTensor(labels), torch.LongTensor(atts), torch.LongTensor(matrix)

            
class Rel_DataSet(Dataset):
    
    def __init__(self, config):
        super(Rel_DataSet, self).__init__()
        training_data = np.load(config.data_path).item()
        self.data_x = training_data['training_x']
        self.data_y = training_data['training_y']
        self.data_matrix = training_data['training_matrix']
        self.tokenizer = BertTokenizer(config.vocab_path, do_lower_case=False)
        self.mat_size = config.max_rel
        self.label_dict = config.label_dict
        self.own = 0 if config.own_rel else -1
        self.tokenize()
        
    def __len__(self):
        return len(self.training_x)
    
        
    def tokenize(self):
        self.training_x, self.training_y, self.training_matrix = list(), list(), list()
        for sentence, labels, mat in zip(self.data_x, self.data_y, self.data_matrix):
            new_sentence, new_label = [], []
            if not sentence:
                continue
            if self.mat_size == 5:
                pass
            else:
                if mat[0,self.mat_size] != -1:
                    continue
            for word, label in zip(sentence, labels):
                tokens = self.tokenizer.tokenize(word)
                tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)
                new_sentence += tokens_id
                new_label += [int(label)] + [self.label_dict['T-concept']]*(len(tokens)-1)
            self.training_x.append(new_sentence)
            self.training_y.append(new_label)
            temp_matrix = mat[:self.mat_size, :self.mat_size]
            self.training_matrix.append(temp_matrix[np.tril_indices(self.mat_size, self.own)]+1)
    
    def __getitem__(self, idx):
        return self.training_x[idx], self.training_y[idx], self.training_matrix[idx]
    
