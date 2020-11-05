import sys
import os
import argparse
import numpy as np
import h5py
import torch
from transformers import *
from modules.transformer_for_nli import *

def component(word1, word2):
	word1 =  word1/np.linalg.norm(word1); word2 = word2/np.linalg.norm(word2)
	w = (word1 - word2)
	w = w/np.linalg.norm(w)
	return w

def get_vec(opt, tokenizer, model):
	words = [opt.wordpiece1, opt.wordpiece2]
	toks = []
	for w in words:
		ts = tokenizer.tokenize(' '+w)
		if len(ts) > 1:
			print('skipping word {0} since it has multiple subtokens.'.format(w))
			continue
		toks.append(ts[0])	# only take the first token/wordpiece

	print('using subtoks:')
	print(toks)
	tok_idx = tokenizer.convert_tokens_to_ids(toks)
	tok_idx = torch.from_numpy(np.asarray(tok_idx))

	emb = model.embeddings.word_embeddings(tok_idx)
	emb = emb.data.cpu()
	assert(emb.shape == (2, 768))

	a = emb[0].view(-1)
	b = emb[1].view(-1)
	dot = torch.dot(a, b)
	a_norm = torch.sqrt(torch.dot(a, a))
	b_norm = torch.sqrt(torch.dot(b, b))
	cos_sim = dot / a_norm / b_norm
	print('cos sim: {0}'.format(cos_sim))

	comp = component(emb[0].numpy(), emb[1].numpy())
	return tok_idx.numpy(), comp


def process(opt):
	model = AutoModel.from_pretrained(opt.transformer_type)
	if isinstance(model, TransformerForNLI):
		model = model.encoder.transformer
		tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
	else:
		tokenizer = AutoTokenizer.from_pretrained(opt.transformer_type)

	tok_idx, comp = get_vec(opt, tokenizer, model)
	print(comp[:10])

	f = h5py.File(opt.output, "w")	
	f['tok_idx'] = tok_idx	
	f["bias"] = comp
	f.close()

	
def main(arguments):
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--wordpiece1', help="word1", default = "he")
	parser.add_argument('--wordpiece2', help="word2", default = "she")
	parser.add_argument('--transformer_type', help="The type of transformer or the pre-trained model of TransformerForNLI", default = "roberta-base")
	parser.add_argument('--output', help="Prefix of the output file names. ", type=str, default = "")
	opt = parser.parse_args(arguments)

	process(opt)

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
