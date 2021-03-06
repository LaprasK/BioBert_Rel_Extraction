import math
import time
import torch
import numpy as np
from collections import OrderedDict
from torch.optim import Adam
from pytorch_pretrained_bert import BertConfig, BertAdam
from model import Rel_Net
from tqdm.auto import tqdm
from metric import f1_score




class build_model(object):
    def __init__(self, config, rel_state_dict_path=None):
        """ner_state_dict is whole weight of BertForTokenClassification, which contains BERT weight + classifier weight,
        if it is not None, then pretrained bert weight will not be loaded.
        """
        self.config = config
        bert_conf = BertConfig(config.bert_conf_path)
        if rel_state_dict_path is None:
            tmp_d = torch.load(config.bert_weight_path, map_location='cpu')
            
            if len(tmp_d.keys()) > 201:
                bert_state_dict = OrderedDict()
                for i in list(tmp_d.keys())[:199]:
                    x = i
                    if i.find('bert') > -1:
                        x = '.'.join(i.split('.')[1:])
                    bert_state_dict[x] = tmp_d[i]
            else:
                bert_state_dict = tmp_d
        else:
            bert_state_dict = None
        
        self.model = Rel_Net(bert_conf, bert_state_dict=bert_state_dict, number_class=8)
        if rel_state_dict_path is not None:
            self.load(rel_state_dict_path)
            
        if config.tune_bert:
            for param in self.model.bert.parameters():
                param.requires_grad = True
        else:
            for param in self.model.bert.parameters():
                param.requires_grad = False
    
        
        self.optimizer = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if torch.cuda.is_available():
            self.model.cuda()

        self.total_epoch = 0
        self.best_f1score = -0.1
        self.train_loss = []
        self.val_loss = []
        self.f1score = []
        self.recall = []
        self.prec = []

    def train(self, train_dataloader, val_dataloader, epochs, tags, save_weight_path=None, start_save=3):
        if self.optimizer is None:
            lr = self.config.lr
        else:
            lr = self.optimizer.get_lr()
        self.optimizer = BertAdam(self.model.parameters(), lr=lr, warmup=self.config.lr_warmup, \
                                  t_total=len(train_dataloader) * epochs)
        total_ep = epochs + self.total_epoch

        ebatches = len(train_dataloader) // 10
        for i in range(self.total_epoch, total_ep):
            print('* [Epoch {}/{}]'.format(i+1, total_ep))
            start_time = time.time()
            self.model.train()
            with tqdm(total=len(train_dataloader), desc="Trainning", \
                      bar_format="{l_bar}{bar} [ time left: {remaining} ]", leave=False) as pbar:
                for step, batch in enumerate(train_dataloader):
                    # add batch to gpu
                    batch = tuple(t.to(self.device) for t in batch)
                    b_input_ids, b_labels, b_input_mask, b_relmatrix = batch

                    self.optimizer.zero_grad()

                    # forward pass
                    loss = self.model(b_input_ids, token_type_ids=None, input_mask=b_input_mask, labels=b_labels, )
                    # backward pass
                    loss.backward()
                    # clip grad
                    #torch.nn.utils.clip_grad_norm_(parameters=self.b_model.parameters(), max_norm=max_grad_norm)
                    # update parameters
                    self.optimizer.step()
                    #self.model.zero_grad()
                    pbar.update(1)
                    if step % ebatches == 0:
                        pbar.write("Step [{}/{}] train loss: {}".format(step, len(train_dataloader), loss.item()))

            print('- Time elasped: {:.5f} seconds\n'.format(time.time() - start_time))
            # VALIDATION on validation set
            print('========== * Evaluating * ===========')
            ret_dic = self.predict(val_dataloader, tags)

            print("Validation loss: {}".format(ret_dic['loss']))
            print("Validation precision: {}".format(ret_dic['prec']))
            print("F1-Score: {}".format(ret_dic['f1-score']))
            
            if ret_dic['f1-score'] > self.best_f1score:
                self.best_f1score = ret_dic['f1-score']
                if save_weight_path is not None and i - total_ep + epochs + 1 >= start_save:
                    print('Saving weight...\n')
                    self.save(save_weight_path)
            self.train_loss.append(loss.item())
            self.val_loss.append(ret_dic['loss'])
            self.f1score.append(ret_dic['f1-score'])
            self.recall.append(ret_dic['recall'])
            self.prec.append(ret_dic['prec'])

            self.total_epoch += 1

    def predict(self, test_dataloader, tags):
        self.model.eval()
        eval_loss, nb_eval_steps = 0, 0
        predictions , true_labels = [], []
        ret_dic = {}
        for batch in test_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_labels, b_input_mask, b_relmatrix = batch
            
            with torch.no_grad():
                tmp_eval_loss = self.model(b_input_ids, token_type_ids=None,
                                    input_mask=b_input_mask, labels=b_labels)
                logits = self.model(b_input_ids, token_type_ids=None,
                            input_mask=b_input_mask)

            mask = b_input_mask.to('cpu').numpy().astype(bool)
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            pred = np.argmax(logits, axis=2)
            # don't count I-<P>
            mask[label_ids == 7] = False
            predictions.extend([p[m] for p, m in zip(pred, mask)])

            true_labels.extend([l[m] for l, m in zip(label_ids, mask)])
            
            eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1
        eval_loss = eval_loss/nb_eval_steps
        pred_tags = [[tags[pi] for pi in p] for p in predictions]
        valid_tags = [[tags[li] for li in l] for l in true_labels]

        ret_dic['loss'] = eval_loss
        ret_dic['prec'], ret_dic['recall'], ret_dic['f1-score'] = f1_score(valid_tags, pred_tags)
        ret_dic['pred'] = pred_tags
        ret_dic['truth'] = valid_tags
        
        return ret_dic
    
    def save(self, path):
        torch.save(self.b_model.state_dict(), path)
    def save_hist(self, path):
        if path is not None:
            hist = np.array([self.train_loss, self.val_loss, self.prec, self.recall, self.f1score])
            np.savetxt(path, hist.T, header='train loss, val loss, precision, recall, f1-score')
    def load(self, path):
        self.b_model.load_state_dict(torch.load(path, map_location='cpu'))