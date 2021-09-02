import sys
import argparse
import h5py
import numpy as np
import torch
from torch import nn
from torch import cuda
from util.holder import *
from util.util import *
from util.data import *
from transformers import *
from modules.multiclass_loss import *
from modules.transformer_for_nli import *
from demo import *

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--load_file', help="Path to where model to be loaded.", default="")

parser.add_argument('--dir', help="Path to the data dir", default="./data/snli_1.0/")
parser.add_argument('--data', help="Path to training data hdf5 file.", default="snli.test.hdf5")
parser.add_argument('--res', help="Path to validation resource files, seperated by comma.", default="")
parser.add_argument('--gpuid', help="The GPU index, if -1 then use CPU", type=int, default=-1)
parser.add_argument('--fp16', help="Whether to use half precision float for speedup", type=int, default=1)
#
parser.add_argument('--pred_output', help="Path to the prediction dump location", default="")
# reference architecture and options which will overwrite the loaded ones
parser.add_argument('--ref_enc', help="Reference encoder type", default="")
parser.add_argument('--ref_cls', help="Reference classifier type", default="")
parser.add_argument('--ref_log', help="Reference log type", default='')
parser.add_argument('--ref_emb_overwrite', help="Reference embeddings to overwrite from", default='')


def fix_opt(opt):
	if opt.ref_enc != '':
		opt.enc = opt.ref_enc
	if opt.ref_cls != '':
		opt.cls = opt.ref_cls
	if opt.ref_log != '':
		opt.log = opt.ref_log

	if opt.ref_emb_overwrite != '':
		opt.emb_overwrite = opt.dir + opt.ref_emb_overwrite
	else:
		opt.emb_overwrite = opt.dir
	return opt


def evaluate(opt, shared, m, data):
	m.train(False)
	shared.is_train = False

	num_ex = 0

	val_idx, val_num_ex = data.subsample(1.0)
	data_size = val_idx.size()[0]
	print('evaluating on {0} batches {1} examples'.format(data_size, val_num_ex))

	loss = MulticlassLoss(opt)

	m.begin_pass(shared)
	loss.begin_pass(shared)
	for i in range(data_size):

		(p_toks, h_toks, label), batch_context = data[i]
		p_toks = Variable(p_toks, requires_grad=False)
		h_toks = Variable(h_toks, requires_grad=False)
		label = Variable(label, requires_grad=False)

		update_shared_context(shared, batch_context)

		with torch.no_grad():
			# forward
			pred = m(p_toks, h_toks)

		batch_loss = loss(pred, label)

		# stats
		num_ex += shared.batch_l

		if (i+1) % 2000 == 0:
			print('evaluated {0} batches'.format(i+1))

	perf, extra_perf = loss.get_epoch_metric()
	loss.end_pass()
	m.end_pass()

	print('finished evaluation on {0} examples'.format(num_ex))

	return (perf, extra_perf)


def main(args):
	shared = Holder()

	opt = parser.parse_args(args)
	opt = fix_opt(opt)

	opt, m, tokenizer = init(opt)

	# 
	opt.data = opt.dir + opt.data
	opt.res = '' if opt.res == ''  else ','.join([opt.dir + path for path in opt.res.split(',')])
	data = Data(opt, opt.data, None if opt.res == '' else opt.res.split(','))

	#
	perf, extra_perf = evaluate(opt, shared, m, data)
	extra_perf_str = ' '.join(['{:.4f}'.format(p) for p in extra_perf])
	print('Val {0:.4f} Extra {1}'.format(
		perf, extra_perf_str))


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))