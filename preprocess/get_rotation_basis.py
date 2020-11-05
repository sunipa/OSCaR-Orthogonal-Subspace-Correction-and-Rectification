import sys
import argparse
import numpy as np
from sklearn.decomposition import PCA
import h5py
import torch
from transformers import *
from modules.transformer_for_nli import *

def component(word1, word2):
	word1 =  word1/np.linalg.norm(word1); word2 = word2/np.linalg.norm(word2)
	w = (word1 - word2)
	w = w/np.linalg.norm(w)
	return w

def get_he_she_basis(opt, tokenizer, model):
	toks = []
	words = ['he', 'she']
	for w in words:
		ts = tokenizer.tokenize(' '+w)
		if len(ts) > 1:
			print('skipping word {0} since it has multiple subtokens.'.format(w))
			continue
		toks.append(ts[0])	# only take the first token/wordpiece

	tok_idx = tokenizer.convert_tokens_to_ids(toks)
	tok_idx = torch.from_numpy(np.asarray(tok_idx))

	emb = model.embeddings.word_embeddings(tok_idx)
	emb = emb.data.cpu()
	assert(emb.shape == (2, 768))

	he = emb[0].view(-1).numpy()
	she = emb[1].view(-1).numpy()

	basis = he - she / np.linalg.norm(he - she)
	return tok_idx.numpy(), basis

def get_basis(opt, tokenizer, model):
	words = opt.wordpieces.split(',')

	if len(words) == 2 and 'he' in words and 'she' in words:
		return get_he_she_basis(opt, tokenizer, model)

	toks = []
	for w in words:
		ts = tokenizer.tokenize(' '+w)
		print(ts)
		if len(ts) > 1:
			print('skipping word {0} since it has multiple subtokens.'.format(w))
			continue
		toks.append(ts[0])	# only take the first token/wordpiece

	tok_idx = tokenizer.convert_tokens_to_ids(toks)
	tok_idx = torch.from_numpy(np.asarray(tok_idx))
	emb = model.embeddings.word_embeddings(tok_idx)
	emb = emb.data.cpu()
	assert(len(emb.shape) == 2)

	pca = PCA(n_components=len(toks))
	pca.fit(emb)
	direction_vector = pca.components_[0]
	return tok_idx.numpy(), direction_vector


def process(opt):
	model = AutoModel.from_pretrained(opt.transformer_type)

	if isinstance(model, TransformerForNLI):
		model = model.encoder.transformer
		tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
	else:
		tokenizer = AutoTokenizer.from_pretrained(opt.transformer_type)

	tok_idx, comp = get_basis(opt, tokenizer, model)
	print(comp[:10])

	print("writing to ", opt.output)
	f = h5py.File(opt.output, "w")		
	f['tok_idx'] = tok_idx
	f["bias"] = comp
	f.close()

	
def main(arguments):
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--wordpieces', help="List of word pieces, separated by comma", default = "")
	parser.add_argument('--sent_template', help="A template used to construct a sentence and get word piece embeddings. <mask> will be replaced by each wordpiece.", 
		default = "A <mask> bought a car.")
	parser.add_argument('--transformer_type', help="The type of transformer or the pre-trained model of TransformerForNLI", default = "roberta-base")
	parser.add_argument('--output', help="Prefix of the output file names. ", type=str, default = "")
	opt = parser.parse_args(arguments)

	process(opt)

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
