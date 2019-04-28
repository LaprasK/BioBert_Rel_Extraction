import os
import argparse
import time
import numpy as np
from model import model
from torch.utils.data import DataLoader
from util import Config, pad_batch, Rel_DataSet

if __name__ == "__main__":
    config = Config('./rel_config.json')
    dataset = Rel_DataSet(config)
    train_len = int(dataset.__len__()*config.training_ratio)
    val_len = dataset.__len__()- train_len
    training, validation = random_split(dataset, [train_len, val_len])
    train_dataloader = DataLoader(dataset=training, batch_size=config.batch_size, shuffle=True,  collate_fn=pad_batch)
    val_dataloader = DataLoader(dataset=validation, batch_size=config.batch_size, shuffle=False, collate_fn=pad_batch)
    
    args = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args.add_argument('--epochs', '-ep', dest='epochs', default=None, type=int,
                         help='epochs')
    args.add_argument('--batch-size', '-bs', dest='batch_size', default=None, type=int,
                         help='batch size')
    args.add_argument('--lr', dest='lr', default=None, type=float,
                         help='lr')
    args.add_argument('--warmup', dest='lr_warmup', default=None, type=float,
                         help='lr warmup, to potion of total steps')
    args.add_argument('--save', '-s', dest='save', action='store_true',
                         help='save the best weight')
    args.add_argument('--start_save', '-ss', dest='start_save', default=2, type=int,
                         help='start save model after these epochs')
    args.add_argument('--rel-weight', '-rel', dest='rel_weight', default=None,
                         help='The rel model weight, which contains bert weight and classifier weight')                      

    pargs = args.parse_args()
    if pargs.epochs is not None:
        config.num_epochs = pargs.epochs
    if pargs.batch_size is not None:
        config.batch_size = pargs.batch_size
    if pargs.lr is not None:
        config.lr = pargs.lr
    if pargs.lr_warmup is not None:
        config.lr_warmup = pargs.lr_warmup
    if pargs.ner_weight is not None:
        config.ner_weight = pargs.ner_weight
        config.bert_weight_path = 'None'
    if pargs.save:
        if not os.path.exists('model_save'): os.makedirs('model_save')
        save_path = os.path.join('model_save', time.strftime("%m-%d-%H-%M-%S", time.localtime()))
        os.makedirs(save_path)
        config.save(os.path.join(save_path, 'model_conf.json'))
        weight_path = os.path.join(save_path, 'ner_weight')
        hist_path = os.path.join(save_path, 'history.txt')
    else:
        weight_path = None
        hist_path = None
    

    # new model
    net = build_model(config, ner_state_dict_path=pargs.ner_weight)
    
    # train
    net.train(train_dataloader, test_dataloader, config.num_epochs, config.idx2tag, save_weight_path=weight_path, start_save=pargs.start_save)
    net.save_hist(hist_path)
    
