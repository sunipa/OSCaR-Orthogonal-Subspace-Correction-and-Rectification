import sys
import os
import ujson
import sys
import argparse
import re
from transformers import *
import torch

def dump(opt):
	tokenizer = AutoTokenizer.from_pretrained(opt.transformer_type)
	vocab = tokenizer.get_vocab()

	model = AutoModel.from_pretrained(opt.transformer_type)

	emb = model.embeddings.word_embeddings.weight.data
	print(len(vocab))
	print(emb.shape)

	with open(opt.output, 'w') as f:
		if opt.output_header == 1:
			f.write('{0} {1}\n'.format(len(vocab), emb.shape[1]))
		for i, (wp, idx) in enumerate(vocab.items()):
			vec = emb[idx].view(-1)
			end = '\n' if i != len(vocab)-1 else ''
			f.write('{0} {1}{2}'.format(wp, ' '.join([str(p.item()) for p in vec]), end))
	print('dumped to {}'.format(opt.output))


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--transformer_type', help="The type of bert encoder from huggingface, eg. roberta-base",default = "roberta-base")
parser.add_argument('--output', help="Prefix to the path of output", default="data/transformer_debias_rotation/roberta_emb.txt")
parser.add_argument('--output_header', help="Whether to output header: vocab_size dim.", type=int, default=0)


def main(args):
	opt = parser.parse_args(args)
	dump(opt)

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))


