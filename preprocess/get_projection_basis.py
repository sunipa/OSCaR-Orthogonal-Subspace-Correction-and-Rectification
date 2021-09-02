import sys
import os
import argparse
import numpy as np
import h5py
import torch
from transformers import *

def component(word1, word2):
	word1 =  word1/np.linalg.norm(word1); word2 = word2/np.linalg.norm(word2)
	w = (word1 - word2)
	w = w/np.linalg.norm(w)
	return w

def get_vec(embeddings, tok_idx):
	assert(len(tok_idx) == 2)

	emb = embeddings(tok_idx)
	emb = emb.data.cpu()

	comp = component(emb[0].numpy(), emb[1].numpy())
	return tok_idx.cpu().numpy(), comp
