import sys
import os
import argparse
import numpy as np
import h5py
import itertools
from collections import defaultdict
import json
import torch
from torch import cuda
from transformers import *
from util.util import *


class Indexer:
	def __init__(self, symbols = ["<blank>"], num_oov=0):
		self.num_oov = num_oov

		self.d = {}
		self.cnt = {}
		for s in symbols:
			self.d[s] = len(self.d)
			self.cnt[s] = 0
			
		for i in range(self.num_oov): #hash oov words to one of 100 random embeddings
			oov_word = '<oov'+ str(i) + '>'
			self.d[oov_word] = len(self.d)
			self.cnt[oov_word] = 10000000	# have a large number for oov word to avoid being pruned
			
	def convert(self, w):		
		return self.d[w] if w in self.d else self.d['<oov' + str(np.random.randint(self.num_oov)) + '>']

	def convert_sequence(self, ls):
		return [self.convert(l) for l in ls]

	def write(self, outfile, with_cnt=True):
		print(len(self.d), len(self.cnt))
		assert(len(self.d) == len(self.cnt))
		with open(outfile, 'w+') as f:
			items = [(v, k) for k, v in self.d.items()]
			items.sort()
			for v, k in items:
				if with_cnt:
					f.write('{0} {1} {2}\n'.format(k, v, self.cnt[k]))
				else:
					f.write('{0} {1}\n'.format(k, v))

	# register tokens only appear in wv
	#   NOTE, only do counting on training set
	def register_words(self, wv, seq, count):
		for w in seq:
			if w in wv and w not in self.d:
				self.d[w] = len(self.d)
				self.cnt[w] = 0
			if w in self.cnt:
				self.cnt[w] = self.cnt[w] + 1 if count else self.cnt[w]

	#   NOTE, only do counting on training set
	def register_all_words(self, seq, count):
		for w in seq:
			if w not in self.d:
				self.d[w] = len(self.d)
				self.cnt[w] = 0
			if w in self.cnt:
				self.cnt[w] = self.cnt[w] + 1 if count else self.cnt[w]

			
def pad(ls, length, symbol, pad_back = True):
	if len(ls) >= length:
		return ls[:length]
	if pad_back:
		return ls + [symbol] * (length -len(ls))
	else:
		return [symbol] * (length -len(ls)) + ls		


def convert(opt, tokenizer, label_indexer, sent1, sent2, label, output):
	np.random.seed(opt.seed)
	num_ex = len(label)

	max_seq_l = opt.max_seq_l + 1 #add 1 for BOS
	targets = np.zeros((num_ex, max_seq_l), dtype=int)
	sources = np.zeros((num_ex, max_seq_l), dtype=int)
	labels = np.zeros((num_ex,), dtype =int)
	source_lengths = np.zeros((num_ex,), dtype=int)
	target_lengths = np.zeros((num_ex,), dtype=int)
	ex_idx = np.zeros(num_ex, dtype=int)
	batch_keys = np.array([None for _ in range(num_ex)])
	
	ex_id = 0
	for _, (src_orig, targ_orig, label_orig) in enumerate(zip(sent1, sent2, label)):
		targ_orig =  targ_orig.strip().split()
		src_orig =  src_orig.strip().split()
		label = label_orig.strip()
		
		sources[ex_id, :len(src_orig)] = np.asarray(tokenizer.convert_tokens_to_ids(src_orig), dtype=int)
		targets[ex_id, :len(targ_orig)] = np.asarray(tokenizer.convert_tokens_to_ids(targ_orig), dtype=int)
		source_lengths[ex_id] = len(src_orig)
		target_lengths[ex_id] = len(targ_orig)
		labels[ex_id] = label_indexer.d[label]
		batch_keys[ex_id] = (source_lengths[ex_id], target_lengths[ex_id])
		ex_id += 1
		if ex_id % 100000 == 0:
			print("{}/{} sentences processed".format(ex_id, num_ex))
	
	print(ex_id, num_ex)
	if opt.shuffle == 1:
		rand_idx = np.random.permutation(ex_id)
		targets = targets[rand_idx]
		sources = sources[rand_idx]
		source_lengths = source_lengths[rand_idx]
		target_lengths = target_lengths[rand_idx]
		labels = labels[rand_idx]
		batch_keys = batch_keys[rand_idx]
		ex_idx = rand_idx
	
	# break up batches based on source/target lengths
	sorted_keys = sorted([(i, p) for i, p in enumerate(batch_keys)], key=lambda x: x[1])
	sorted_idx = [i for i, _ in sorted_keys]
	# rearrange examples	
	sources = sources[sorted_idx]
	targets = targets[sorted_idx]
	labels = labels[sorted_idx]
	target_l = target_lengths[sorted_idx]
	source_l = source_lengths[sorted_idx]
	ex_idx = rand_idx[sorted_idx]
	
	curr_l_src = 0
	curr_l_targ = 0
	batch_location = [] #idx where sent length changes
	for j,i in enumerate(sorted_idx):
		if batch_keys[i][0] != curr_l_src or batch_keys[i][1] != curr_l_targ:
			curr_l_src = source_lengths[i]
			curr_l_targ = target_lengths[i]
			batch_location.append(j)
	if batch_location[-1] != len(sources): 
		batch_location.append(len(sources)-1)
	
	#get batch sizes
	curr_idx = 0
	batch_idx = [0]
	for i in range(len(batch_location)-1):
		end_location = batch_location[i+1]
		while curr_idx < end_location:
			curr_idx = min(curr_idx + opt.batch_size, end_location)
			batch_idx.append(curr_idx)

	batch_l = []
	target_l_new = []
	source_l_new = []
	for i in range(len(batch_idx)):
		end = batch_idx[i+1] if i < len(batch_idx)-1 else len(sources)
		batch_l.append(end - batch_idx[i])
		source_l_new.append(source_l[batch_idx[i]])
		target_l_new.append(target_l[batch_idx[i]])
		
		# sanity check
		for k in range(batch_idx[i], end):
			assert(source_l[k] == source_l_new[-1])
			assert(sources[k, source_l[k]:].sum() == 0)

	
	# Write output
	f = h5py.File(output, "w")		
	f["source"] = sources
	f["target"] = targets
	f["label"] = labels
	f["target_l"] = np.array(target_l_new, dtype=int)
	f["source_l"] = np.array(source_l_new, dtype=int)
	f["batch_l"] = batch_l
	f["batch_idx"] = batch_idx
	f['ex_idx'] = ex_idx
	print("saved {} batches ".format(len(f["batch_l"])))
	f.close()  

def tokenize_and_write(tokenizer, sent_ls, output):
	bos_tok, eos_tok = get_special_tokens(tokenizer)

	all_tokenized = []
	for sent in sent_ls:
		toks = tokenizer.tokenize(sent)
		toks = [bos_tok] + toks + [eos_tok]
		all_tokenized.append(' '.join(toks))

	print('writing tokenized to {0}'.format(output))
	with open(output, 'w') as f:
		for seq in all_tokenized:
			f.write(seq + '\n')

	return all_tokenized


def extract(csv_file):
	all_sent1 = []
	all_sent2 = []
	all_label = []

	skip_cnt = 0

	with open(csv_file, 'r') as f:
		line_idx = 0
		for l in f:
			line_idx += 1
			if line_idx == 1 or l.strip() == '':
				continue

			cells = l.rstrip().split('\t')
			label = cells[0].strip()
			sent1 = cells[5].strip()
			sent2 = cells[6].strip()

			if label == '-':
				#print('skipping label {0}'.format(label))
				skip_cnt += 1
				continue

			assert(label in ['entailment', 'neutral', 'contradiction'])

			all_sent1.append(sent1)
			all_sent2.append(sent2)
			all_label.append(label)

	print('skipped {0} examples'.format(skip_cnt))

	return (all_sent1, all_sent2, all_label)


def process(opt):
	tokenizer_output = opt.tokenizer_output+'.' if opt.tokenizer_output != opt.dir else ''

	label_indexer = Indexer(symbols=["entailment", "neutral", "contradiction"], num_oov=0)

	tokenizer = AutoTokenizer.from_pretrained(opt.transformer_type, add_special_tokens=False)

	#### extract, tokenize, and record
	print('tokenizing ', opt.train)
	sent1, sent2, label = extract(opt.train)
	sent1_train = tokenize_and_write(tokenizer, sent1, tokenizer_output + 'train.sent1.txt')
	sent2_train = tokenize_and_write(tokenizer, sent2, tokenizer_output + 'train.sent2.txt')
	label_train = label

	print('tokenizing ', opt.val)
	sent1, sent2, label = extract(opt.val)
	sent1_val = tokenize_and_write(tokenizer, sent1, tokenizer_output + 'dev.sent1.txt')
	sent2_val = tokenize_and_write(tokenizer, sent2, tokenizer_output + 'dev.sent2.txt')
	label_val = label

	print('tokenizing ', opt.test)
	sent1, sent2, label = extract(opt.test)
	sent1_test = tokenize_and_write(tokenizer, sent1, tokenizer_output + 'test.sent1.txt')
	sent2_test = tokenize_and_write(tokenizer, sent2, tokenizer_output + 'test.sent2.txt')
	label_test = label

	label_indexer.write(opt.output + ".label.dict")
	assert(len(label_indexer.d) == 3)

	# batch up
	convert(opt, tokenizer, label_indexer, sent1_train, sent2_train, label_train, opt.output + ".train.hdf5")
	convert(opt, tokenizer, label_indexer, sent1_val, sent2_val, label_val, opt.output + ".val.hdf5")
	convert(opt, tokenizer, label_indexer, sent1_test, sent2_test, label_test, opt.output + ".test.hdf5")

	
def main(arguments):
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--train', help="Path to training data csv file.", default = "snli_1.0_train.txt")
	parser.add_argument('--val', help="Path to validation data csv file.", default = "snli_1.0_dev.txt")
	parser.add_argument('--test', help="Path to test data csv file.", default = "snli_1.0_test.txt")
	parser.add_argument('--dir', help="Path to the data dir",default = "./data/snli_1.0/")
	
	parser.add_argument('--transformer_type', help="The type of bert encoder from huggingface, eg. roberta-base",default = "roberta-base")

	parser.add_argument('--batch_size', help="Maximal size of each minibatch, actual batches will be likely smaller than this.", type=int, default=36)
	parser.add_argument('--max_seq_l', help="Maximum sequence length. Sequences longer than this are dropped.", type=int, default=400)
	parser.add_argument('--tokenizer_output', help="Prefix of the tokenized output file names. ", type=str, default = "snli")
	parser.add_argument('--output', help="Prefix of the output file names. ", type=str, default = "snli")
	parser.add_argument('--shuffle', help="If = 1, shuffle sentences before sorting (based on source length).", type = int, default = 1)
	parser.add_argument('--seed', help="The random seed", type = int, default = 1)
	opt = parser.parse_args(arguments)

	opt.train = opt.dir + opt.train
	opt.val = opt.dir + opt.val
	opt.test = opt.dir + opt.test
	opt.output = opt.dir + opt.output
	opt.tokenizer_output = opt.dir + opt.tokenizer_output

	process(opt)

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
