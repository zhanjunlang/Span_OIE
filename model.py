# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 19:33:57 2018

@author: win 10
"""

import torch
import torch.nn as nn
from utils import to_var
import torch.nn.functional as F

class Pred_finder(nn.Module):
    def __init__(self,glove, label_size, pos_size, pos_dim, hidden_dim, n_layers):
        super(Pred_finder, self).__init__()
        embedding_matrix = glove.get_embedding_matrix()
        self.vocab_size = embedding_matrix.shape[0]
        self.word_dim = embedding_matrix.shape[1]        
        self.word_emb = nn.Embedding(self.vocab_size, self.word_dim)
        self.word_emb.weight.data.copy_(torch.from_numpy(embedding_matrix))        
        self.pos_emb = nn.Embedding(pos_size, pos_dim)
        self.pos_dim = pos_dim
        self.layers = n_layers
        self.bilstm = nn.LSTM(input_size=self.word_dim+pos_dim,
                              hidden_size=hidden_dim,
                              num_layers = n_layers,
                              bidirectional=True,
                              dropout = 0.2)
        self.pred_classifier = nn.Linear(8*hidden_dim, 2) 
        
    def get_LSTM_features(self,word_input, pos_input):
        word_embedded = self.word_emb(word_input)
        pos_embedded = self.pos_emb(pos_input)
        input_ = torch.cat([word_embedded, pos_embedded], dim=1).view(-1,1,self.word_dim+self.pos_dim)
        out, _ = self.bilstm(input_) 
        return out #(seq_len, 1, 2*hidden_size)
       
    def get_span_features(self, lstm_features, candidates):
        span_features = []
        for term in candidates:
            start = lstm_features[term[0]]
            end = lstm_features[term[1]]
            span_feature = torch.cat([start, end, start+end, start-end],dim=1) #(1, 4*hidden_size)
            span_features.append(span_feature)
        span_features = torch.cat(span_features,dim=0)
        return span_features
    
    def predicate_prediction(self, lstm_features, pred_candidates):
        span_features = self.get_span_features(lstm_features, pred_candidates)
        out = self.pred_classifier(span_features) 
        out = F.log_softmax(out, dim=1)
        return out #(span_num,2)
    
    
    def forward(self, word_input, pos_input, pred_candidates): 
        lstm_features = self.get_LSTM_features(word_input, pos_input)        
        pred_out = self.predicate_prediction(lstm_features, pred_candidates)
        return pred_out

        
class Span_labler(nn.Module):
    def __init__(self,glove, label_size, pos_size, pos_dim, hidden_dim, n_layers, dp_size, dp_dim, syntax_flag):
        super(Span_labler, self).__init__()
        embedding_matrix = glove.get_embedding_matrix()
        self.vocab_size = embedding_matrix.shape[0]
        self.word_dim = embedding_matrix.shape[1]        
        self.word_emb = nn.Embedding(self.vocab_size, self.word_dim)
        self.word_emb.weight.data.copy_(torch.from_numpy(embedding_matrix))        
        self.pos_emb = nn.Embedding(pos_size, pos_dim)
        self.pos_dim = pos_dim
        self.position_dim = 5
        self.position_emb = nn.Embedding(2, self.position_dim) 
        self.dp_emb = nn.Embedding(dp_size, dp_dim)
        self.dp_dim = dp_dim
        self.layers = n_layers
        self.label_size = label_size
        self.syntax_flag = syntax_flag
        self.bilstm = nn.LSTM(input_size=self.word_dim+pos_dim+self.position_dim+self.dp_dim,
                              hidden_size=hidden_dim,
                              num_layers = n_layers,
                              bidirectional=True,
                              dropout = 0.3)
                               
        if self.syntax_flag:
            self.lin_r = nn.Sequential(
                    nn.Linear(10*hidden_dim+dp_dim, 5*hidden_dim),
                    nn.Dropout(0.3),
                    nn.ReLU(),
                    nn.Linear(5*hidden_dim, label_size, bias=False),
                    )
        else:
            self.lin_r = nn.Sequential(
                    nn.Linear(8*hidden_dim, 4*hidden_dim),
                    nn.Dropout(0.3),
                    nn.ReLU(),
                    nn.Linear(4*hidden_dim, label_size, bias=False),
                    )
        
    def get_LSTM_features(self,word_input, pos_input, dp_input, target):
        position_input = []
        for i in range(len(word_input)):
            if i < target[0] or i > target[1]:
                position_input.append(0)
            else:
                position_input.append(1)
        position_input = to_var(torch.LongTensor(position_input))
        word_embedded = self.word_emb(word_input)
        pos_embedded = self.pos_emb(pos_input)
        dp_embedded = self.dp_emb(dp_input)
        position_embedded = self.position_emb(position_input)
        input_ = torch.cat([word_embedded, pos_embedded, dp_embedded, position_embedded], dim=1).view(-1,1,self.word_dim+self.pos_dim+self.dp_dim+self.position_dim)
        out, _ = self.bilstm(input_) 
        
        return out #(seq_len, 1, 2*hidden_size)
    
    def get_span_features(self, lstm_features, candidates, candidates_head, candidates_head_dp):
        span_features = []
        for i, term in enumerate(candidates):
            start = lstm_features[term[0]]
            end = lstm_features[term[1]]
            if self.syntax_flag:
                head = lstm_features[candidates_head[i]]
                head_dp_emb = self.dp_emb(to_var(torch.LongTensor([candidates_head_dp[i]])))
                span_feature = torch.cat([start, end, start+end,start-end, head, head_dp_emb],dim=1) #(1, 10*hidden_size+dp_dim)
            else:
                span_feature = torch.cat([start, end, start+end,start-end],dim=1) #(1, 8*hidden_size)
            span_features.append(span_feature)
        span_features = torch.cat(span_features,dim=0)
        return span_features
        
    def argument_prediction(self, lstm_features, arg_candidates, candidates_head, candidates_head_dp, gold_pred_idx):
        span_features = self.get_span_features(lstm_features, arg_candidates, candidates_head, candidates_head_dp)
        #pred_feature = span_features[gold_pred_idx].view(1,-1)
        #cat_features = torch.cat([span_features,pred_feature.expand(span_features.size()[0],pred_feature.size()[1])],dim=1)
        score_r = self.lin_r(span_features) #(span_num, label_num)
        score = F.log_softmax(score_r, dim=0)
        return score
        
    def forward(self, word_input, pos_input, dp_input, arg_candidates, candidates_head, candidates_head_dp, gold_pred_idx): 
        lstm_features = self.get_LSTM_features(word_input, pos_input, dp_input, arg_candidates[gold_pred_idx])        
        arg_out = self.argument_prediction(lstm_features, arg_candidates, candidates_head, candidates_head_dp, gold_pred_idx)
        return arg_out
        
        
        
        
        
        
        
        
        