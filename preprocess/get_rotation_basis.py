import sys
import argparse
import numpy as np
from sklearn.decomposition import PCA
import h5py
import torch
from transformers import *

def get_he_she_basis(embeddings, he_she_idx):
	emb = embeddings(he_she_idx)
	emb = emb.data.cpu()
	assert(emb.shape == (2, 768))

	he = emb[0].view(-1).numpy()
	she = emb[1].view(-1).numpy()

	basis = (he - she) / np.linalg.norm(he - she)
	return he_she_idx.cpu().numpy(), basis


def get_basis(embeddings, tok_idx):
	emb = embeddings(tok_idx)
	emb = emb.data.cpu()
	assert(len(emb.shape) == 2)

	pca = PCA(n_components=len(tok_idx))
	pca.fit(emb)
	direction_vector = pca.components_[0]
	return tok_idx.cpu().numpy(), direction_vector

