import sys
import argparse
import h5py
import numpy as np
import torch
from torch import nn
from torch import cuda
from util.holder import *
from util.util import *
from preprocess.preprocess import pad
from transformers import *
from modules.transformer_for_nli import *
import traceback


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--load_file', help="The path to pretrained model (optional)", default = "")
parser.add_argument('--gpuid', help="The GPU index, if -1 then use CPU", type=int, default=-1)
parser.add_argument('--fp16', help="Whether to use half precision float for speedup", type=int, default=1)

def process(opt, tokenizer, p, h):
	bos_tok, eos_tok = get_special_tokens(tokenizer)

	p_toks = tokenizer.tokenize(p)
	p_toks = [bos_tok] + p_toks + [eos_tok]
	p_toks = np.array(tokenizer.convert_tokens_to_ids(p_toks), dtype=int)

	h_toks = tokenizer.tokenize(h)
	h_toks = [bos_tok] + h_toks + [eos_tok]
	h_toks = np.array(tokenizer.convert_tokens_to_ids(h_toks), dtype=int)

	context = Holder()
	context.batch_l = 1
	context.source_l = np.asarray([len(p_toks)])
	context.target_l = np.asarray([len(h_toks)])
	context.data_name = ""
	context.res_map = {}
	context.batch_ex_idx = [0]

	return (p_toks, h_toks), context


def pretty_print_pred(opt, pred):
	y = pred.view(-1).argmax(-1)
	log = opt.labels[y]
	return log

def run(opt, m, tokenizer, seq, predicates=[]):
	shared = Holder()

	(p_toks, h_toks), batch_context = process(opt, tokenizer, seq, predicates)

	update_shared_context(shared, batch_context)

	p_toks = to_device(Variable(torch.tensor([p_toks]), requires_grad=False), opt.gpuid)
	h_toks = to_device(Variable(torch.tensor([h_toks]), requires_grad=False), opt.gpuid)

	with torch.no_grad():
		pred = m(p_toks, h_toks)

	log = pretty_print_pred(opt, pred)
	return log


def fix_opt(opt):
	# if opt is loaded as argparse.Namespace from json, it would lose track of data type, enforce types here
	opt.label_map_inv = {int(k): v for k, v in opt.label_map_inv.items()}
	opt.num_label = len(opt.labels)
	opt.dropout = 0
	return opt


def init(opt):
	m = AutoModel.from_pretrained(opt.load_file, global_opt=opt)
	tokenizer = AutoTokenizer.from_pretrained(opt.transformer_type, add_special_tokens=False)
	opt = fix_opt(m.config)

	if opt.gpuid != -1:
		m.distribute()

	if opt.fp16 == 1:
		m = m.half()

	return opt, m, tokenizer


def main(args):
	opt = parser.parse_args(args)
	opt, m, tokenizer = init(opt)

	p = "A man holding a cup of coffee is walking on the street."
	h = "Somebody is moving."
	log = run(opt, m, tokenizer, p, h)

	print('###################################')
	print('Here is a sample prediction for input:')
	print('>> P: ', p)
	print('>> H: ', h)
	print(log)

	print('###################################')
	print('#           Instructions          #')
	print('###################################')
	print('>> Enter a premise (P) and hypothesis (H) sentences as prompted.')

	while True:
		try:
			print('###################################')
			p = input("Enter P: ")
			h = input("Enter H: ")

			log = run(opt, m, tokenizer, p, h)
			print(log)

		except KeyboardInterrupt:
			return
		except BaseException as e:
			traceback.print_tb(e.__traceback__)


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))

