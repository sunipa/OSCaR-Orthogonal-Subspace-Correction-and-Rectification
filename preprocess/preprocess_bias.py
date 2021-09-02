# Preprocessing unlabeled NLI exampples for OSCaR
import sys
import argparse
from util.util import *
from transformers import *
from .preprocess import Indexer, tokenize_and_write, convert


def write_to(ls, out_file):
	print('writing to {0}'.format(out_file))
	with open(out_file, 'w+') as f:
		for l in ls:
			f.write((l + '\n'))

def extract_oscar(csv_file):
	all_sent1 = []
	all_sent2 = []
	all_x_pairs = []	# the pair of words (x1, x2) in sentence template

	firstline = ''
	with open(csv_file, 'r') as f:
		firstline = f.readline()

	with open(csv_file, 'r') as f:
		line_idx = 0
		# if reading from generated files
		if firstline.startswith('id,pair type,premise_filler_word'):
			for l in f:
				line_idx += 1
				if line_idx == 1 or l.strip() == '':
					continue
	
				cells = l.rstrip().split(',')
				x1 = cells[2]
				x2 = cells[3]
				sent1 = cells[-2]
				sent2 = cells[-1]
				
				all_x_pairs.append((x1, x2))
				all_sent1.append(sent1)
				all_sent2.append(sent2)
		# if reading from model dumps
		elif firstline.startswith('x1,x2,premise'):
			for l in f:
				line_idx += 1
				if line_idx == 1 or l.strip() == '':
					continue
				cells = l.rstrip().split(',')
				x1 = cells[0]
				x2 = cells[1]
				sent1 = cells[2]
				sent2 = cells[3]

				all_x_pairs.append((x1, x2))
				all_sent1.append(sent1)
				all_sent2.append(sent2)
		else:
			raise Exception('unrecognized file format, first line: ', firstline)

	return all_x_pairs, all_sent1, all_sent2


def process_oscar(opt):
	label_indexer = Indexer(symbols=["entailment", "neutral", "contradiction"], num_oov=0)

	tokenizer = AutoTokenizer.from_pretrained(opt.transformer_type, add_special_tokens=False)

	#### extract, tokenize, and record
	print('tokenizing ', opt.data)
	x_pairs, sent1, sent2 = extract_oscar(opt.data)
	write_to(['{0} {1}'.format(x1, x2) for x1, x2 in x_pairs], opt.output + '.x_pair.txt')
	write_to(sent1, opt.output + '.raw.sent1.txt')
	write_to(sent2, opt.output + '.raw.sent2.txt')
	sent1_toks = tokenize_and_write(tokenizer, sent1, opt.output + '.sent1.txt')
	sent2_toks = tokenize_and_write(tokenizer, sent2, opt.output + '.sent2.txt')
	gold_label = 'neutral' if opt.gold_label == "" else opt.gold_label
	label = [gold_label for _ in range(len(sent1))]	# fake labels

	# batch up
	convert(opt, tokenizer, label_indexer, sent1_toks, sent2_toks, label, opt.output + ".hdf5")

	
def main(arguments):
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--data', help="Path to the unlabeled data in csv format.", default = "")
	parser.add_argument('--dir', help="Path to the data dir",default = "./data/oscar/")
	
	parser.add_argument('--transformer_type', help="The type of bert encoder from huggingface, eg. roberta-base",default = "roberta-base")
	parser.add_argument('--gold_label', help="The gold label to fix to for all examples, default is empty, falling back to neutral", default = "")

	parser.add_argument('--batch_size', help="Maximal size of each minibatch, actual batches will be likely smaller than this.", type=int, default=48)
	parser.add_argument('--max_seq_l', help="Maximum sequence length. Sequences longer than this are dropped.", type=int, default=400)
	parser.add_argument('--output', help="Prefix of the output file names. ", type=str, default = "")
	parser.add_argument('--shuffle', help="If = 1, shuffle sentences before sorting (based on source length).", type = int, default = 1)
	parser.add_argument('--seed', help="The random seed", type = int, default = 1)
	opt = parser.parse_args(arguments)

	opt.data = opt.dir + opt.data
	opt.output = opt.dir + opt.output

	process_oscar(opt)

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))


