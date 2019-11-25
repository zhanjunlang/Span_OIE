# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 15:15:23 2018

@author: win 10
"""

import torch
import json
from torch.autograd import Variable

cuda_num = 0
def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda(cuda_num)
    return Variable(x, volatile=volatile)

def make_span_candidates(l):
        candidates = []
        for i in range(l):
            for j in range(i,l):
                candidates.append([i,j])
        return candidates

arg_BIO_labels = {
        "A0-B":0,
        "A0-I":1,
        "A1-B":2,
        "A1-I":3,
        "A2-B":4,
        "A2-I":5,
        "A3-B":6,
        "A3-I":7,
        "O":8
        }  

pred_BIO_labels={
        "P-B":0,
        "P-I":1,
        "O":2
        } 
NLTK_POS_TAGS = [
    '$',
    '\'\'',
    '(',
    ')',
    ',',
    '--',
    '.',
    'CC',
    'CD',
    'DT',
    'EX',
    'FW',
    'IN',
    'JJ',
    'JJR',
    'JJS',
    'LS',
    'MD',
    'NN',
    'NNP',
    'NNPS',
    'NNS',
    'PDT',
    'POS',
    'PRP',
    'PRP$',
    'RB',
    'RBR',
    'RBS',
    'RP',
    'SYM',
    'TO',
    'UH',
    'VB',
    'VBD',
    'VBG',
    'VBN',
    'VBP',
    'VBZ',
    'WDT',
    'WP',
    'WP$',
    'WRB',
    '``',
    ':',
    '#',
]

SPACY_POS_TAGS = [
    "<UNK>",
    "-LRB-",
    "-RRB-",
    ",",
    ":",
    ".",
    "''",
    "\"\"",
    "#",
    "``",
    "$",
    "ADD",
    "AFX",
    "BES",
    "CC",
    "CD",
    "DT",
    "EX",
    "FW",
    "GW",
    "HVS",
    "HYPH",
    "IN",
    "JJ",
    "JJR",
    "JJS",
    "LS",
    "MD",
    "NFP",
    "NIL",
    "NN",
    "NNP",
    "NNPS",
    "NNS",
    "PDT",
    "POS",
    "PRP",
    "PRP$",
    "RB",
    "RBR",
    "RBS",
    "RP",
    "_SP",
    "SYM",
    "TO",
    "UH",
    "VB",
    "VBD",
    "VBG",
    "VBN",
    "VBP",
    "VBZ",
    "WDT",
    "WP",
    "WP$",
    "WRB",
    "XX",
]
if __name__ == "__main__":
    pos2index = {}
    for sym in SPACY_POS_TAGS:
        if sym not in pos2index:
            pos2index[sym] = len(pos2index)
    
    with open("data/pos2index.json","w") as f:
        json.dump(pos2index,f)
    