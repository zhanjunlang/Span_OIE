#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 12:43:54 2018

@author: longzhan
"""


import numpy as np
import logging
from sys import version_info

logging.basicConfig(level = logging.DEBUG)

UNK_SYMBOL = "<unk>"
UNK_INDEX = 0
UNK_VALUE = lambda dim: np.zeros(dim) # get an UNK of a specificed dimension

class Glove:
    """
    Stores pretrained word embeddings for GloVe, and
    outputs a Keras Embeddings layer.
    """
    def __init__(self, fn, dim = None):
        """
        Load a GloVe pretrained embeddings model.
        fn - Filename from which to load the embeddings
        dim - Dimension of expected word embeddings, used as verficiation,
              None avoids this check.
        """
        self.fn = fn
        self.dim = dim
        logging.debug("Loading GloVe embeddings from: {} ...".format(self.fn))
        self._load(self.fn)
        logging.debug("Done!")

    def _load(self, fn):
        """
        Load glove embedding from a given filename
        """
        self.word_index = {UNK_SYMBOL : UNK_INDEX}
        emb = []
        if version_info.major == 3:
            for line in open(fn,encoding='utf-8'):
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                if self.dim:
                    assert(len(coefs) == self.dim)
                else:
                    self.dim = len(coefs)
    
                # Record mapping from word to index
                self.word_index[word] = len(emb) + 1
                emb.append(coefs)
        else:
            for line in open(fn):
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                if self.dim:
                    assert(len(coefs) == self.dim)
                else:
                    self.dim = len(coefs)
    
                # Record mapping from word to index
                self.word_index[word] = len(emb) + 1
                emb.append(coefs)

        # Add UNK at the first index in the table
        self.emb = np.array([UNK_VALUE(self.dim)] + emb)
        # Set the vobabulary size
        self.vocab_size = len(self.emb)

    def get_word_index(self, word, lower = True):
        """
        Get the index of a given word (int).
        If word doesnt exists, returns UNK.
        lower - controls whether the word should be lowered before checking map
        """
        if lower:
            word = word.lower()
        return self.word_index[word] \
            if (word in self.word_index) else UNK_INDEX

    def get_embedding_matrix(self):
        return self.emb

    