import sys
import argparse
import h5py
import os
import random
import time
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch import cuda
from util.holder import *
from util.util import *
from util.data import *
from modules.multiclass_loss import *
from modules.optimizer import *
from modules.transformer_for_nli import *
from log.unlabeled_log import *

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dir', help="Path to the data dir", default="data/snli_1.0/")
parser.add_argument('--train_data', help="Path to training data hdf5 file.", default="snli.train.hdf5")
parser.add_argument('--val_data', help="Path to validation data hdf5 file.", default="snli.val.hdf5")
parser.add_argument('--save_file', help="Path to where model to be saved.", default="model")
parser.add_argument('--load_file', help="The path to pretrained model (optional)", default = "")
parser.add_argument('--label_dict', help="The path to label dictionary", default = "snli.label.dict")
# resource specs
parser.add_argument('--train_res', help="Path to training resource files, seperated by comma.", default="")
parser.add_argument('--val_res', help="Path to validation resource files, seperated by comma.", default="")
## pipeline specs
parser.add_argument('--dropout', help="The dropout probability", type=float, default=0.1)
parser.add_argument('--percent', help="The percent of training data to use", type=float, default=1.0)
parser.add_argument('--epochs', help="The number of epoches for training", type=int, default=5)
parser.add_argument('--optim', help="The name of optimizer to use for training", default='adamw_apex')
parser.add_argument('--learning_rate', help="The learning rate for training", type=float, default=0.00003)
parser.add_argument('--clip', help="The norm2 threshold to clip, set it to negative to disable", type=float, default=1.0)
parser.add_argument('--adam_betas', help="The betas used in adam", default='0.9,0.999')
parser.add_argument('--weight_decay', help="The factor of weight decay", type=float, default=0.01)
# bert specs
parser.add_argument('--transformer_type', help="The type of bert encoder from huggingface, eg. roberta-base",default = "roberta-base")
parser.add_argument('--warmup_perc', help="The percentages of total expectec updates to warmup", type=float, default=0.1)
parser.add_argument('--warmup', help="The type of warmup function", default='linear')
## pipeline stages
parser.add_argument('--enc', help="The type of encoder", default='transformer')
parser.add_argument('--cls', help="The type of classifier", default='linear')
parser.add_argument('--loss', help="The type of loss", default='multiclass')
parser.add_argument('--log', help="The type of log", default='')
#
parser.add_argument('--param_init_type', help="The type of parameter initialization", default='xavier_normal')
parser.add_argument('--print_every', help="Print stats after this many batches", type=int, default=500)
parser.add_argument('--seed', help="The random seed", type=int, default=3435)
parser.add_argument('--gpuid', help="The GPU index, if -1 then use CPU", type=int, default=-1)
parser.add_argument('--acc_batch_size', help="The accumulative batch size, -1 to disable", type=int, default=-1)
# adhoc options
parser.add_argument('--freeze_emb', help="Whether to freeze transformer and only update classifier", type=int, default=0)
parser.add_argument('--emb_overwrite', help="Path to emb txt file which will be loaded and overwrite the transformer word_embeddings", default="")
# oscar related options
parser.add_argument('--bias_update_every', help="Number of gradient updates every bias update happens between, -1: static (update only once", type=int, default=-1)


def complete_opt(opt):
	if 'base' in opt.transformer_type:
		opt.hidden_size = 768
	elif 'large' in opt.transformer_type:
		opt.hidden_size = 1024
	#
	if hasattr(opt, 'train_data'): 
		opt.train_data = opt.dir + opt.train_data
	if hasattr(opt, 'val_data'):
		opt.val_data = opt.dir + opt.val_data
	if hasattr(opt, 'train_res'):
		opt.train_res = '' if opt.train_res == ''  else ','.join([opt.dir + path for path in opt.train_res.split(',')])
	if hasattr(opt, 'val_res'):
		opt.val_res = '' if opt.val_res == ''  else ','.join([opt.dir + path for path in opt.val_res.split(',')])
	if hasattr(opt, 'emb_overwrite'):
		opt.emb_overwrite = opt.dir + opt.emb_overwrite

	if hasattr(opt, 'label_dict'):
		opt.label_dict = opt.dir + opt.label_dict
		opt.labels, opt.label_map_inv = load_label_dict(opt.label_dict)

	# if opt is loaded as argparse.Namespace from json, it would lose track of data type, enforce types here
	opt.label_map_inv = {int(k): v for k, v in opt.label_map_inv.items()}
	opt.num_label = len(opt.labels)

	# default on transformers pretrained config
	config = AutoConfig.from_pretrained(opt.transformer_type)
	_opt = TransformerForNLIConfig()
	_opt.__dict__.update(config.__dict__)
	_opt.__dict__.update(opt.__dict__)
	_opt.architectures = 'TransformerForNLI'
	_opt.model_type = 'transformerfornli'
	return _opt

# train batch by batch, accumulate batches until the size reaches acc_batch_size
def train_epoch(opt, shared, m, optim, data, sub_idx):
	train_loss = 0.0
	num_ex = 0
	num_batch = 0
	acc_batch_size = 0
	start_time = time.time()
	shared.is_train = True

	# subsamples of data
	# if subsample indices provided, permutate from subsamples
	#	else permutate from all the data
	data_size = sub_idx.size()[0]
	batch_order = torch.randperm(data_size)
	batch_order = sub_idx[batch_order]
	all_data = []
	for i in range(data_size):
		all_data.append((data, batch_order[i]))

	loss = MulticlassLoss(opt)

	m.train(True)
	optim.begin_pass(shared)
	m.begin_pass(shared)
	loss.begin_pass(shared)
	for i in range(data_size):

		cur_data, cur_idx = all_data[i]
		(p_toks, h_toks, label), batch_context = cur_data[cur_idx]

		p_toks = Variable(p_toks, requires_grad=False)
		h_toks = Variable(h_toks, requires_grad=False)
		label = Variable(label, requires_grad=False)

		# setup context for the current batch
		update_shared_context(shared, batch_context)

		# forward
		pred = m(p_toks, h_toks)
		batch_loss = loss(pred, label)

		# stats
		train_loss += float(batch_loss.item())
		num_ex += shared.batch_l
		num_batch += 1
		acc_batch_size += shared.batch_l

		# accumulate grads
		#optim.backward(m, batch_loss/shared.batch_l)
		optim.backward(m, batch_loss)	# this performs better somehow

		if shared.num_update == 0:
			grad_sanity_check(optim, m)

		# accumulate current batch until the rolled up batch size exceeds threshold or meet certain boundary
		if i == data_size-1 or acc_batch_size >= opt.acc_batch_size or (i+1) % opt.print_every == 0:

			optim.step(m)

			shared.num_update += 1
			acc_batch_size = 0

			# clear up grad
			m.zero_grad()

			# stats
			time_taken = time.time() - start_time

			if (i+1) % opt.print_every == 0:
				stats = '{0}, Batch {1:.1f}k '.format(shared.epoch+1, float(i+1)/1000)
				stats += 'Grad {0:.1f}/{1:.1f} '.format(optim.min_grad_norm2, optim.max_grad_norm2)
				stats += 'Loss {0:.4f} '.format(train_loss / num_ex)
				stats += loss.print_cur_stats()
				stats += ' Time {0:.1f}'.format(time_taken)
				print(stats)

	loss.end_pass()
	optim.end_pass()
	m.end_pass()

	perf, extra_perf = loss.get_epoch_metric()

	return perf, extra_perf

def train(opt, shared, m, optim, train_data, val_data):
	best_val_perf = -1.0	# something < 0
	test_perf = 0.0
	train_perfs = []
	val_perfs = []
	extra_perfs = []

	print('{0} batches in train set'.format(train_data.size()))

	train_idx, train_num_ex = train_data.subsample(opt.percent)
	print('{0} examples sampled for training'.format(train_num_ex))
	print('for the record, the first 10 training batches are: {0}'.format(train_idx[:10]))
	
	val_idx, val_num_ex = val_data.subsample(1.0)
	print('{0} examples sampled for dev'.format(val_num_ex))
	print('for the record, the first 10 dev batches are: {0}'.format(val_idx[:10]))

	shared.num_train_ex = train_num_ex
	shared.num_update = 0
	start = 0
	for i in range(start, opt.epochs):
		shared.epoch = i

		train_perf, extra_train_perf = train_epoch(opt, shared, m, optim, train_data, train_idx)
		train_perfs.append(train_perf)
		extra_perf_str = ' '.join(['{:.4f}'.format(p) for p in extra_train_perf])
		print('Train {0:.4f} All {1}'.format(train_perf, extra_perf_str))

		# evaluate
		#	and save if it's the best model
		val_perf, extra_val_perf = validate(opt, shared, m, val_data, val_idx)
		val_perfs.append(val_perf)
		extra_perfs.append(extra_val_perf)
		extra_perf_str = ' '.join(['{:.4f}'.format(p) for p in extra_val_perf])
		print('Val {0:.4f} All {1}'.format(val_perf, extra_perf_str))

		perf_table_str = ''
		cnt = 0
		print('Epoch  | Train | Val ...')
		for train_perf, extra_perf in zip(train_perfs, extra_perfs):
			extra_perf_str = ' '.join(['{:.4f}'.format(p) for p in extra_perf])
			perf_table_str += '{0}\t{1:.4f}\t{2}\n'.format(cnt+1, train_perf, extra_perf_str)
			cnt += 1
		print(perf_table_str)

		if val_perf > best_val_perf:
			best_val_perf = val_perf
			print('saving model to {0}'.format(opt.save_file))

			tokenizer = AutoTokenizer.from_pretrained(opt.transformer_type, add_special_tokens=False)
			tokenizer.save_pretrained(opt.save_file)
			m.save_pretrained(opt.save_file)

		else:
			print('skip saving model for perf <= {0:.4f}'.format(best_val_perf))



def validate(opt, shared, m, val_data, val_idx):
	m.train(False)
	shared.is_train = False

	val_loss = 0.0
	num_ex = 0
	num_batch = 0

	data_size = val_idx.size()[0]
	all_val = []
	for i in range(data_size):
		all_val.append((val_data, val_idx[i]))

	#data_size = val_idx.size()[0]
	print('validating on the {0} batches...'.format(data_size))

	loss = MulticlassLoss(opt)

	m.begin_pass(shared)
	loss.begin_pass(shared)
	for i in range(data_size):
		cur_data, cur_idx = all_val[i]
		(p_toks, h_toks, label), batch_context = cur_data[cur_idx]

		p_toks = Variable(p_toks, requires_grad=False)
		h_toks = Variable(h_toks, requires_grad=False)
		label = Variable(label, requires_grad=False)

		# setup context for the current batch
		update_shared_context(shared, batch_context)

		with torch.no_grad():
			# forward
			pred = m(p_toks, h_toks)
		batch_loss = loss(pred, label)

		# stats
		val_loss += float(batch_loss.item())
		num_ex += shared.batch_l
		num_batch += 1

	perf, extra_perf = loss.get_epoch_metric()	# we only use the first loss's corresponding metric to select models
	loss.end_pass()
	m.end_pass()

	return perf, extra_perf


def main(args):
	opt = parser.parse_args(args)
	shared = Holder()

	opt = complete_opt(opt)

	torch.manual_seed(opt.seed)
	if opt.gpuid != -1:
		torch.cuda.set_device(opt.gpuid)
		torch.cuda.manual_seed_all(opt.seed)

	# initializing from pretrained
	if opt.load_file != '':		
		m = TransformerForNLI.from_pretrained(opt.load_file, global_opt = opt)

	else:
		m = TransformerForNLI(opt)
		m.init_weight()

	print(opt)
	
	if opt.gpuid != -1:
		m.distribute()	# distribute to multigpu

	optim = get_optimizer(opt)
	m = optim.build_optimizer(m)	# build optimizer after distributing model to devices

	# loading data
	train_data = Data(opt, opt.train_data, None if opt.train_res == '' else opt.train_res.split(','))
	val_data = Data(opt, opt.val_data, None if opt.val_res == '' else opt.val_res.split(','))

	train(opt, shared, m, optim, train_data, val_data)



if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))