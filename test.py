# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 14:44:01 2018

@author: win 10
"""

from model import Span_labler,Pred_finder
from load_pretrained_embedding import Glove
import json
import torch
from utils import make_span_candidates, to_var, cuda_num
import spacy
from main_span import pred_span_candidates_filter, arg_span_candidates_filter, get_span_head
from main_span import hidden_dim, label_size, n_layers, pos_dim, dp_dim

syntax_flag = True
model_fn = "model/model.params_span_last_"+str(syntax_flag)+"_e0"
pred_model_fn = "model/pred_model.params_span_False"

def prepare_sentence(sent_spacy,glove,pos2index,dp2index):
    arg_candidates = make_span_candidates(len(sent_spacy))
    pred_candidates = pred_span_candidates_filter(sent_spacy, arg_candidates,5)  
    sent_idx = [glove.get_word_index(word.text) for word in sent_spacy]
    pos_idx = [pos2index[word.tag_] for word in sent_spacy]
    dp_idx = []
    for word in sent_spacy:
        if word.dep_ in dp2index:
            dp_idx.append(dp2index[word.dep_])
        else:
            dp_idx.append(dp2index["<UNK>"])
    word_input = to_var(torch.LongTensor(sent_idx))
    pos_input = to_var(torch.LongTensor(pos_idx))
    dp_input = to_var(torch.LongTensor(dp_idx))
    return word_input, pos_input, dp_input, pred_candidates, arg_candidates, sent_spacy

def is_overlap(span1,span2): 
    start = [span1[0],span2[0]]
    end = [span1[1],span2[1]]
    if min(end)<max(start):
        return False
    else:
        return True
   
if __name__ == "__main__":
    nlp = spacy.load("en_core_web_md")
    glove = Glove("data/glove.6B.100d.txt")
    with open("data/pos2index.json") as f:
        pos2index = json.load(f)
        pos_size = len(pos2index)
    with open("data/dp2index.json") as f:
        dp2index = json.load(f)
    model = Span_labler(glove, label_size, pos_size, pos_dim, hidden_dim, n_layers, len(dp2index), dp_dim, syntax_flag)
    pred_model = Pred_finder(glove, label_size, pos_size, pos_dim, hidden_dim, n_layers)
    if torch.cuda.is_available():
        model = model.cuda(cuda_num)
        pred_model = pred_model.cuda(cuda_num)
    model.load_state_dict(torch.load(model_fn))
    model.eval()
    pred_model.load_state_dict(torch.load(pred_model_fn))
    pred_model.eval()
 
    sentence = "she holds a bachelor of commerce degree , obtained from the university of nairobi ."
    sent_spacy = nlp(sentence)
    word_input, pos_input, dp_input, pred_candidates, arg_candidates, sent_spacy = prepare_sentence(sent_spacy, glove, pos2index, dp2index)
    #找出谓词
    pred_out = pred_model(word_input, pos_input, pred_candidates)
    _, max_index = torch.max(pred_out,dim=1)
    gold_pred_all = []
    for i,v in enumerate(max_index):
        if v.data.item() == 1:
            pair = pred_candidates[i]
            gold_pred_all.append(pair)
            
    print("---")
    
    new_gold_pred_all = []
    mark = [0 for t in gold_pred_all]
    for pred_id, gold_pred in enumerate(gold_pred_all):
        if mark[pred_id] == 0:
            combine_flag = False
            for j in range(len(gold_pred_all)):
                if mark[j] == 0:
                    if gold_pred[1] + 1 == gold_pred_all[j][0]:
                        new_gold_pred_all.append([gold_pred[0], gold_pred_all[j][1]])
                        mark[pred_id] = 1
                        mark[j] = 1
                        combine_flag = True
                        break
                    if gold_pred[0] - 1 == gold_pred_all[j][1]:
                        new_gold_pred_all.append([gold_pred_all[j][0],gold_pred[1]])
                        mark[pred_id] = 1
                        mark[j] = 1
                        combine_flag = True
                        break
            if combine_flag == False and gold_pred not in new_gold_pred_all:
                new_gold_pred_all.append(gold_pred)
                mark[pred_id] = 1
                        
    for gold_pred in new_gold_pred_all:
        do_flag = True
        for pair in new_gold_pred_all:
            if pair != gold_pred:
                if (pair[0] <= gold_pred[0]) and (pair[1] >= gold_pred[1]):
                    do_flag = False
                    print("skip")
                    break
        if do_flag == False:
            continue
        
        print(sent_spacy[gold_pred[0]:gold_pred[1]+1].text)
        fill = [0,0,0,0]
        fill_span = [[-1,-1],[-1,-1],[-1,-1],[-1,-1]]
        arg_candidates = arg_span_candidates_filter(sent_spacy,arg_candidates,500,gold_pred)
        candidates_head, candidates_head_dp = get_span_head(sent_spacy, arg_candidates, dp2index)
        gold_pred_idx = arg_candidates.index(gold_pred)
        arg_out = model(word_input, pos_input, dp_input, arg_candidates, candidates_head, candidates_head_dp, gold_pred_idx)
        arg_out = arg_out.transpose(0,1) #(num_label, num_span)
        arg_out = arg_out.cpu().detach().numpy()
        flatten = {}
        for i in range(arg_out.shape[0]):
            for j in range(arg_out.shape[1]):
                flatten[i+10*j] = arg_out[i][j]
        flatten_sorted = sorted(flatten.items(), key=lambda item:item[1], reverse=True)
        for pair in flatten_sorted:
            if fill == [1,1,1,1]:
                break
            else:
                label = int(pair[0] % 10)
                span_idx = int(pair[0] / 10)
                if fill[label] == 1:
                    continue
                elif span_idx == gold_pred_idx:
                    fill[label] = 1
                else:
                    span = arg_candidates[span_idx]
                    add_flag = True
                    for selected_span in fill_span:
                        if is_overlap(span, selected_span):
                            add_flag = False
                            break
                    if label == 0:
                        if span[0] >= gold_pred[0]:
                            add_flag = False
                    if label == 1:
                        if span[0] <= gold_pred[0]:
                            add_flag = False
                    if add_flag:
                        fill[label] = 1
                        fill_span[label] = span
                        print("arg",str(label),": ",sent_spacy[span[0]:span[1]+1].text)
        print("-----")
            
            
    