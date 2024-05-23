# -*- coding: utf-8 -*-
# file: infer_example.py
# author: songyouwei <youwei0314@gmail.com>
# fixed: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2019. All Rights Reserved.

import torch
import torch.nn.functional as F
import argparse
import numpy as np

from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, pad_and_truncate
from models import LSTM, IAN, MemNet, RAM, TD_LSTM, TC_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN, ASGCN, LCF_BERT
from models.aen import CrossEntropyLoss_LSR, AEN_BERT
from models.bert_spc import BERT_SPC
from dependency_graph import dependency_adj_matrix

from transformers import BertModel, AutoModel

class Inferer:
    """A simple inference example"""
    def __init__(self, opt):
        self.opt = opt
        if 'bert' in opt.model_name:
            self.tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = AutoModel.from_pretrained(opt.pretrained_bert_name, return_dict=False)
            self.model = opt.model_class(bert, opt).to(opt.device)
        else:
            self.tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_seq_len=opt.max_seq_len,
                dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
            embedding_matrix = build_embedding_matrix(
                word2idx=self.tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
            self.model = opt.model_class(embedding_matrix, opt)
        print('loading model {0} ...'.format(opt.model_name))
        self.model.load_state_dict(torch.load(opt.state_dict_path))
        self.model = self.model.to(opt.device)
        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    def evaluate(self, text, aspect):
        aspect = aspect.lower().strip()
        text_left, _, text_right = [s.strip() for s in text.lower().partition(aspect)]
        
        text_indices = self.tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
        context_indices = self.tokenizer.text_to_sequence(text_left + " " + text_right)
        left_indices = self.tokenizer.text_to_sequence(text_left)
        left_with_aspect_indices = self.tokenizer.text_to_sequence(text_left + " " + aspect)
        right_indices = self.tokenizer.text_to_sequence(text_right, reverse=True)
        right_with_aspect_indices = self.tokenizer.text_to_sequence(aspect + " " + text_right, reverse=True)
        aspect_indices = self.tokenizer.text_to_sequence(aspect)
        left_len = np.sum(left_indices != 0)
        aspect_len = np.sum(aspect_indices != 0)
        aspect_boundary = np.asarray([left_len, left_len + aspect_len - 1], dtype=np.int64)

        text_len = np.sum(text_indices != 0)
        concat_bert_indices = self.tokenizer.text_to_sequence('[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
        concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
        concat_segments_indices = pad_and_truncate(concat_segments_indices, self.tokenizer.max_seq_len)

        text_bert_indices = self.tokenizer.text_to_sequence("[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
        aspect_bert_indices = self.tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

        dependency_graph = dependency_adj_matrix(text)

        data = {
            'concat_bert_indices': concat_bert_indices,
            'concat_segments_indices': concat_segments_indices,
            'text_bert_indices': text_bert_indices,
            'aspect_bert_indices': aspect_bert_indices,
            'text_indices': text_indices,
            'context_indices': context_indices,
            'left_indices': left_indices,
            'left_with_aspect_indices': left_with_aspect_indices,
            'right_indices': right_indices,
            'right_with_aspect_indices': right_with_aspect_indices,
            'aspect_indices': aspect_indices,
            'aspect_boundary': aspect_boundary,
            'dependency_graph': dependency_graph,
        }

        t_inputs = [torch.tensor([data[col]], device=self.opt.device) for col in self.opt.inputs_cols]
        t_outputs = self.model(t_inputs)
        t_probs = F.softmax(t_outputs, dim=-1).cpu().numpy()

        return t_probs


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='lstm', type=str, choices=['lstm','td_lstm','tc_lstm','atae_lstm'
        ,'ian','memnet','ram','cabasc','tnet_lf','aoa,mgan','bert_spc','aen_bert','lcf_bert'], help=
    'choose model from lstm,td_lstm,tc_lstm,atae_lstm,ian,memnet,ram,cabasc,tnet_lf,aoa,mgan,bert_spc,aen_bert,lcf_bert')
    parser.add_argument('--dataset', default='ulasan_ori', choices=['twitter', 'acl14shortdata', 'SemEval2014',
                                                                 'SemEval2015', 'SemEval2016', 'twitter_know',
                                                                 'acl14shortdata_know', 'SemEval2014_know',
                                                                 'SemEval2015_know', 'SemEval2016_know',
                                                                 'ulasan_ori', 'ulasan_raw_know',
                                                                 'ulasan_trim_know', 'ulasan_select_know',
                                                                 'padanan_ori', 'padanan_know',
                                                                 'padanan_trim_know', 'padanan_select_know'], type=str,
                        help='choose from twitter, acl14shortdata, SemEval2014, SemEval2015, SemEval2016 |||_know')
    parser.add_argument('--state_dict_path', default='state_dict/bert_spc_ulasan_ori_know_val_f1_0.6543', type=str)
    parser.add_argument('--kalimat', default='permainan di taman air sangat bervariasi, mulai dari yang santai hingga ekstrim. tempat yang sangat cocok untuk liburan keluarga', type=str)
    parser.add_argument('--aspek', default='liburan', type=str)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--lr', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=100, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=100, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=125, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--patience', default=20, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=1234, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float,
                        help='set ratio between 0 and 1 for validation support')
    # The following parameters are only valid for the lcf-bert model
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=3, type=int,
                        help='semantic-relative-distance, see the paper of LCF-BERT model')

    model_classes = {
        'lstm': LSTM,
        'td_lstm': TD_LSTM,
        'tc_lstm': TC_LSTM,
        'atae_lstm': ATAE_LSTM,
        'ian': IAN,
        'memnet': MemNet,
        'ram': RAM,
        'cabasc': Cabasc,
        'tnet_lf': TNet_LF,
       'aoa': AOA,
        'mgan': MGAN,
        'asgcn': ASGCN,
        'bert_spc': BERT_SPC,
        'aen_bert': AEN_BERT,
        'lcf_bert': LCF_BERT,
    }
    dataset_files = {
        'twitter': {
            'train': './datasets/acl-14-short-data/train.raw',
            'test': './datasets/acl-14-short-data/test.raw'
        },
        'restaurant': {
            'train': './datasets/semeval14/Restaurants_Train.xml.seg',
            'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
        },
        'laptop': {
            'train': './datasets/semeval14/Laptops_Train.xml.seg',
            'test': './datasets/semeval14/Laptops_Test_Gold.xml.seg'
        },
        'ulasan_ori': {
            'train': './datasets/ulasan_ori/train.tsv',
            'test': './datasets/ulasan_ori/dev.tsv'
        },
        'ulasan_raw_know': {
            'train': './datasets/ulasan_ori/a_raw_knowledge/train.tsv',
            'test': './datasets/ulasan_ori/a_raw_knowledge/dev.tsv'
        },
        'ulasan_trim_know': {
            'train': './datasets/ulasan_ori/c_trimmed_knowledge/train.tsv',
            'test': './datasets/ulasan_ori/c_trimmed_knowledge/dev.tsv'
        },
        'ulasan_select_know': {
            'train': './datasets/ulasan_ori/d_selected_knowledge/train.tsv',
            'test': './datasets/ulasan_ori/d_selected_knowledge/dev.tsv'
        },
        'padanan_ori': {
            'train': './datasets/ulasan_padanan/train.tsv',
            'test': './datasets/ulasan_padanan/dev.tsv'
        },
        'padanan_know': {
            'train': './datasets/ulasan_padanan/b_padanan_knowledge/train.tsv',
            'test': './datasets/ulasan_padanan/b_padanan_knowledge/dev.tsv'
        },
        'padanan_trim_know': {
            'train': './datasets/ulasan_padanan/c_trimmed_knowledge/train.tsv',
            'test': './datasets/ulasan_padanan/c_trimmed_knowledge/dev.tsv'
        },
        'padanan_select_know': {
            'train': './datasets/ulasan_padanan/d_selected_knowledge/train.tsv',
            'test': './datasets/ulasan_padanan/d_selected_knowledge/dev.tsv'
        }
    }
    input_colses = {
        'lstm': ['text_indices'],
        'td_lstm': ['left_with_aspect_indices', 'right_with_aspect_indices'],
        'tc_lstm': ['left_with_aspect_indices', 'right_with_aspect_indices', 'aspect_indices'],
        'atae_lstm': ['text_indices', 'aspect_indices'],
        'ian': ['text_indices', 'aspect_indices'],
        'memnet': ['context_indices', 'aspect_indices'],
        'ram': ['text_indices', 'aspect_indices', 'left_indices'],
        'cabasc': ['text_indices', 'aspect_indices', 'left_with_aspect_indices', 'right_with_aspect_indices'],
        'tnet_lf': ['text_indices', 'aspect_indices', 'aspect_in_text'],
        'aoa': ['text_indices', 'aspect_indices'],
        'mgan': ['text_indices', 'aspect_indices', 'left_indices'],
        'asgcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
        'bert_spc': ['concat_bert_indices', 'concat_segments_indices'],
        'aen_bert': ['text_bert_indices', 'aspect_bert_indices'],
        'lcf_bert': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices', 'aspect_bert_indices'],
    }
    class Option(object): pass
    opt = Option()
    # opt.model_name = 'bert_spc'
    opt.model_class = model_classes[opt.model_name]
    # opt.dataset = 'ulasan_ori'
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    # set your trained models here
    # opt.state_dict_path = 'state_dict/ian_restaurant_acc0.7911'
    opt.embed_dim = 300
    opt.hidden_dim = 300
    opt.max_seq_len = 125
    opt.bert_dim = 768
    # opt.pretrained_bert_name = 'bert-base-uncased'
    opt.polarities_dim = 3
    opt.hops = 3
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.local_context_focus = 'cdm'
    opt.SRD = 3
    opt.dropout = 0.1

    inf = Inferer(opt)
    # t_probs = inf.evaluate('the service is terrible', 'service')
    t_probs = inf.evaluate(opt.kalimat, opt.aspek)
    print(t_probs.argmax(axis=-1) - 1)
