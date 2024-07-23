# -*- coding: utf-8 -*-
# file: infer_example.py
# author: songyouwei <youwei0314@gmail.com>
# fixed: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2019. All Rights Reserved.

from curses import has_il
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
        },
        'combined_ori': {
            'train': './datasets/ulasan_combined/train.tsv',
            'test': './datasets/ulasan_combined/dev.tsv'
        },
        'combined_raw_know': {
            'train': './datasets/ulasan_combined/a_raw_know/train.tsv',
            'test': './datasets/ulasan_combined/a_raw_know/dev.tsv'
        },
        'combined_padanan_know': {
            'train': './datasets/ulasan_combined/b_padanan_knowledge/train.tsv',
            'test': './datasets/ulasan_combined/b_padanan_knowledge/dev.tsv'
        },
        'combined_padanan_trim': {
            'train': './datasets/ulasan_combined/c_trimmed_knowledge/train.tsv',
            'test': './datasets/ulasan_combined/c_trimmed_knowledge/dev.tsv'
        },
        'combined_padanan_select': {
            'train': './datasets/ulasan_combined/d_selected_knowledge/train.tsv',
            'test': './datasets/ulasan_combined/d_selected_knowledge/dev.tsv'
        },
        'combined_select_know': {
            'train': './datasets/ulasan_combined/e_raw_selected_knowledge/train.tsv',
            'test': './datasets/ulasan_combined/e_raw_selected_knowledge/dev.tsv'
        },
        'combined_trim_know': {
            'train': './datasets/ulasan_combined/f_raw_trimmed_knowledge/train.tsv',
            'test': './datasets/ulasan_combined/f_raw_trimmed_knowledge/dev.tsv'
        },
        'combined_trim_select_up': {
            'train': './datasets/ulasan_combined/g_trimmed_selected_up_knowledge/train.tsv',
            'test': './datasets/ulasan_combined/g_trimmed_selected_up_knowledge/dev.tsv'
        },
        'insert_raw_know': {
            'train': './datasets/ulasan_combined/x_insert_raw_knowledge/train.tsv',
            'test': './datasets/ulasan_combined/x_insert_raw_knowledge/dev.tsv'
        },
        'insert_trim_know': {
            'train': './datasets/ulasan_combined/y_insert_trimmed_knowledge/train.tsv',
            'test': './datasets/ulasan_combined/y_insert_trimmed_knowledge/dev.tsv'
        },
        'insert_select_know': {
            'train': './datasets/ulasan_combined/z_insert_selected_knowledge/train.tsv',
            'test': './datasets/ulasan_combined/z_insert_selected_knowledge/dev.tsv'
        },
        'insert_padanan_know': {
            'train': './datasets/ulasan_combined/j_insert_padanan_knowledge/train.tsv',
            'test': './datasets/ulasan_combined/j_insert_padanan_knowledge/dev.tsv'
        },
        'insert_padanan_trim': {
            'train': './datasets/ulasan_combined/k_insert_padanan_trimmed_knowledge/train.tsv',
            'test': './datasets/ulasan_combined/k_insert_padanan_trimmed_knowledge/dev.tsv'
        },
        'insert_padanan_select': {
            'train': './datasets/ulasan_combined/l_insert_padanan_selected_knowledge/train.tsv',
            'test': './datasets/ulasan_combined/l_insert_padanan_selected_knowledge/dev.tsv'
        },
        'insert_trim_select_up': {
            'train': './datasets/ulasan_combined/m_insert_trim_select_up_knowledge/train.tsv',
            'test': './datasets/ulasan_combined/m_insert_trim_select_up_knowledge/dev.tsv'
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
    opt.model_name = 'bert_spc'
    opt.model_class = model_classes[opt.model_name]
    opt.dataset = 'ulasan_ori'
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    # set your trained models here
    opt.state_dict_path = 'state_dict/bert_spc_ulasan_ori_val_f1_0.7176'
    opt.embed_dim = 300
    opt.hidden_dim = 300
    opt.max_seq_len = 125
    opt.bert_dim = 768
    opt.pretrained_bert_name = 'indobenchmark/indobert-base-p1'
    opt.polarities_dim = 3
    opt.hops = 3
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.local_context_focus = 'cdm'
    opt.SRD = 3
    opt.dropout = 0.1

    # import pandas as pd
    # test_infer = pd.read_csv('./datasets/test/test.tsv', sep='\t', usecols=['review', 'aspect'])
    
    inf = Inferer(opt)
    t_probs = inf.evaluate('pura ini dibangun diatas tebing terjal. pandangan yang sangat indah. di kawasan pura ini banyak monyet yang siap merampok topi, ponsel dan kaca mata anda. hatihatilah dengan mereka.', 'pura')
    # t_probs = test_infer.apply(lambda x: inf.evaluate(x['review'], x['aspect']), axis=1)
    print(t_probs.argmax(axis=-1) - 1)
    # print('x')
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        # print(t_probs)