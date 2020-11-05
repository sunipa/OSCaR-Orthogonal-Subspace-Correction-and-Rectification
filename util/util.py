import sys
import h5py
import torch
from torch import nn
from torch import cuda
import string
import re
from collections import Counter
import numpy as np
from transformers import *

def update_shared_context(shared, context):
	shared.__dict__.update(context.__dict__)

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

	if hasattr(opt, 'label_dict'):
		opt.label_dict = opt.dir + opt.label_dict
		opt.labels, opt.label_map_inv = load_label_dict(opt.label_dict)

	# if opt is loaded as argparse.Namespace from json, it would lose track of data type, enforce types here
	opt.label_map_inv = {int(k): v for k, v in opt.label_map_inv.items()}
	opt.num_label = len(opt.labels)

	# default on transformers pretrained config
	config = AutoConfig.from_pretrained(opt.transformer_type)
	config.__dict__.update(opt.__dict__)
	return config


def load_label_dict(label_dict):
	labels = []
	label_map_inv = {}
	with open(label_dict, 'r') as f:
		for l in f:
			if l.strip() == '':
				continue
			toks = l.rstrip().split()
			labels.append(toks[0])
			label_map_inv[int(toks[1])] = toks[0]
	return labels, label_map_inv
	

def get_special_tokens(tokenizer):
	CLS, SEP = tokenizer.cls_token, tokenizer.sep_token
	if CLS is None or SEP is None:
		CLS, SEP = tokenizer.bos_token, tokenizer.eos_token
	if CLS is None:
		CLS = SEP
	return CLS, SEP

def to_device(x, gpuid):
	if gpuid == -1:
		return x.cpu()
	if x.device != gpuid:
		return x.cuda(gpuid)
	return x

def has_nan(t):
	return torch.isnan(t).sum() == 1

def tensor_on_dev(t, is_cuda):
	if is_cuda:
		return t.cuda()
	else:
		return t

def pick_label(dist):
	return np.argmax(dist, axis=1)

def torch2np(t, is_cuda):
	return t.numpy() if not is_cuda else t.cpu().numpy()

def save_opt(opt, path):
	with open(path, 'w') as f:
		f.write('{0}'.format(opt))


def last_index(ls, key):
	return len(ls) - 1 - ls[::-1].index(key)

def load_param_dict(path):
	# TODO, this is ugly
	f = h5py.File(path, 'r')
	return f


def save_param_dict(param_dict, path):
	file = h5py.File(path, 'w')
	for name, p in param_dict.items():
		file.create_dataset(name, data=p)

	file.close()


def load_dict(path):
	rs = {}
	with open(path, 'r+') as f:
		for l in f:
			if l.strip() == '':
				continue
			w, idx, cnt = l.strip().split()
			rs[int(idx)] = w
	return rs


def rand_tensor(shape, r1, r2):
	return (r1 - r2) * torch.rand(shape) + r2


def max_with_mask(v, dim):
	max_v, max_idx = v.max(dim)
	return max_v, max_idx, torch.zeros(v.shape).to(v).scatter(dim, max_idx.unsqueeze(dim), 1.0)

def min_with_mask(v, dim):
	min_v, min_idx = v.min(dim)
	return min_v, min_idx, torch.zeros(v.shape).to(v).scatter(dim, min_idx.unsqueeze(dim), 1.0)


# use the idx (batch_l, seq_l, rs_l) (2nd dim) to select the middle dim of the content (batch_l, seq_l, d)
#	the result has shape (batch_l, seq_l, rs_l, d)
def batch_index2_select(content, idx, nul_idx):
	idx = idx.long()
	rs_l = idx.shape[-1]
	batch_l, seq_l, d = content.shape
	content = content.contiguous().view(-1, d)
	shift = torch.arange(0, batch_l).to(idx.device).long().view(batch_l, 1, 1)
	shift = shift * seq_l
	shifted = idx + shift
	rs = content[shifted].view(batch_l, seq_l, rs_l, d)
	#
	mask = (idx != nul_idx).unsqueeze(-1)
	return rs * mask.to(rs)

# use the idx (batch_l, rs_l) (1st dim) to select the middle dim of the content (batch_l, seq_l, d)
#	return (batch_l, rs_l, d)
def batch_index1_select(content, idx, nul_idx):
	idx = idx.long()
	rs_l = idx.shape[-1]
	batch_l, seq_l, d = content.shape
	content = content.contiguous().view(-1, d)
	shift = torch.arange(0, batch_l).to(idx.device).long().view(batch_l, 1)
	shift = shift * seq_l
	shifted = idx + shift
	rs = content[shifted].view(batch_l, rs_l, d)
	#
	mask = (idx != nul_idx).unsqueeze(-1)
	return rs * mask.to(rs)