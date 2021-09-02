import sys
import math
import torch
from torch import nn
from torch import cuda
import apex
import transformers
from transformers.optimization import *
from util.holder import *
from util.util import *
import torch.optim as optim


def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 0.5 * (1.0 + math.cos(math.pi * x))

def warmup_constant(x, warmup=0.002):
    """ Linearly increases learning rate over `warmup`*`t_total` (as provided to BertAdam) training steps.
        Learning rate is 1. afterwards. """
    if x < warmup:
        return x/warmup
    return 1.0

def warmup_linear(x, warmup=0.002):
    """ Specifies a triangular learning rate schedule where peak is reached at `warmup`*`t_total`-th (as provided to BertAdam) training step.
        After `t_total`-th training step, learning rate is zero. """
    if x < warmup:
        return x/warmup
    return max((x-1.)/(warmup-1.), 0)


def get_warmup_func(option):
	if option == 'linear':
		return warmup_linear
	elif option == 'constant':
		return warmup_constant
	elif option == 'cosine':
		return warmup_cosine
	elif option == 'no_warmup':
		return lambda x, l: 1.0
	else:
		raise Exception('unrecognized warmup func', option)



# the torch.amp for fp16 with huggingface AdamW
class AdamWAmp:
	def __init__(self, opt):
		self.opt = opt
		self.optim = None
		self.scaler = None

		self.min_grad_norm2 = 1000000000.0
		self.max_grad_norm2 = -1.0
		
	def build_optimizer(self, m, avg_batch_size=10):
		self.avg_batch_size = avg_batch_size
		no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
		named_params = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
		optimizer_grouped_parameters = [{'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': self.opt.weight_decay},
			{'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

		self.optim = transformers.AdamW(optimizer_grouped_parameters, lr=self.opt.learning_rate)
		self.scaler = cuda.amp.GradScaler()
		return m

	def get_lr(self):
		if self.opt.warmup_perc <= 0:
			return self.opt.learning_rate
		acc_l = self.avg_batch_size if self.opt.acc_batch_size < 0 else self.opt.acc_batch_size
		normalizer = self.shared.num_train_ex / acc_l * self.opt.epochs
		warmup_func = get_warmup_func(self.opt.warmup)
		return self.opt.learning_rate * warmup_func(self.shared.num_update / normalizer, self.opt.warmup_perc)

	def step(self, m):
		cur_lr = self.get_lr()
		for param_group in self.optim.param_groups:
			param_group['lr'] = cur_lr

		self.scaler.step(self.optim)
		self.scaler.update()

	# this interface is only for apex's optimizer
	def backward(self, m, loss):
		self.scaler.scale(loss).backward()

		if self.opt.clip > 0:
			self.scaler.unscale_(self.optim)	# This does not work with gradient accumulation (acc_batch_size > -1)
			grad_norm2 = torch.nn.utils.clip_grad_norm_(m.parameters(), self.opt.clip)

		self.min_grad_norm2 = min(self.min_grad_norm2, grad_norm2)
		self.max_grad_norm2 = max(self.max_grad_norm2, grad_norm2)

	def begin_pass(self, shared):
		self.shared = shared
		self.min_grad_norm2 = 1000000000.0
		self.max_grad_norm2 = -1.0

	def end_pass(self):
		pass


# the apex's adam for fp16 with huggingface AdamW
class AdamWApex:
	def __init__(self, opt):
		self.opt = opt
		self.optim = None

		self.min_grad_norm2 = 1000000000.0
		self.max_grad_norm2 = -1.0
		
	def build_optimizer(self, m, avg_batch_size=10):
		self.avg_batch_size = avg_batch_size
		no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
		named_params = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
		optimizer_grouped_parameters = [{'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': self.opt.weight_decay},
			{'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

		adamw = transformers.AdamW(optimizer_grouped_parameters, lr=self.opt.learning_rate)
		m, self.optim = apex.amp.initialize(m, adamw, opt_level='O1')
		return m

	def get_lr(self):
		if self.opt.warmup_perc <= 0:
			return self.opt.learning_rate
		acc_l = self.avg_batch_size if self.opt.acc_batch_size < 0 else self.opt.acc_batch_size
		normalizer = self.shared.num_train_ex / acc_l * self.opt.epochs
		warmup_func = get_warmup_func(self.opt.warmup)
		return self.opt.learning_rate * warmup_func(self.shared.num_update / normalizer, self.opt.warmup_perc)

	def step(self, m):
		cur_lr = self.get_lr()
		for param_group in self.optim.param_groups:
			param_group['lr'] = cur_lr

		self.optim.step()

	# this interface is only for apex's optimizer
	def backward(self, m, loss):
		with apex.amp.scale_loss(loss, self.optim) as scaled_loss:
			scaled_loss.backward()
		grad_norm2 = torch.nn.utils.clip_grad_norm_(apex.amp.master_params(self.optim), self.opt.clip)

		self.min_grad_norm2 = min(self.min_grad_norm2, grad_norm2)
		self.max_grad_norm2 = max(self.max_grad_norm2, grad_norm2)

	def begin_pass(self, shared):
		self.shared = shared
		self.min_grad_norm2 = 1000000000.0
		self.max_grad_norm2 = -1.0

	def end_pass(self):
		pass


class AdamW:
	def __init__(self, opt):
		self.opt = opt
		self.optim = None

		self.min_grad_norm2 = 1000000000.0
		self.max_grad_norm2 = -1.0

	def build_optimizer(self, m, avg_batch_size=10):
		self.avg_batch_size = avg_batch_size
		no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
		named_params = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
		optimizer_grouped_parameters = [{'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': self.opt.weight_decay},
			{'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

		self.optim = transformers.AdamW(optimizer_grouped_parameters, lr=self.opt.learning_rate)
		return m

	def get_lr(self):
		if self.opt.warmup_perc <= 0:
			return self.opt.learning_rate
		acc_l = self.avg_batch_size if self.opt.acc_batch_size < 0 else self.opt.acc_batch_size
		normalizer = self.shared.num_train_ex / acc_l * self.opt.epochs
		warmup_func = get_warmup_func(self.opt.warmup)
		return self.opt.learning_rate * warmup_func(self.shared.num_update / normalizer, self.opt.warmup_perc)

	def step(self, m):
		cur_lr = self.get_lr()
		for param_group in self.optim.param_groups:
			param_group['lr'] = cur_lr

		self.optim.step()

	# this interface is only for apex's optimizer
	def backward(self, m, loss):
		loss.backward()
		grad_norm2 = torch.nn.utils.clip_grad_norm_(m.parameters(), self.opt.clip)

		self.min_grad_norm2 = min(self.min_grad_norm2, grad_norm2)
		self.max_grad_norm2 = max(self.max_grad_norm2, grad_norm2)

	def begin_pass(self, shared):
		self.shared = shared
		self.min_grad_norm2 = 1000000000.0
		self.max_grad_norm2 = -1.0

	def end_pass(self):
		pass



def get_optimizer(opt):
	optim = None
	if opt.optim == 'adamw_apex':
		optim = AdamWApex(opt)
	elif opt.optim == 'adamw_amp':
		optim = AdamWAmp(opt)
	elif opt.optim == 'adamw':
		optim = AdamW(opt)
	else:
		raise Exception('unrecognized optim: {0}'.format(opt.optim))
	return optim


def grad_sanity_check(optim, m):
	for n, p in m.named_parameters():
		if p.grad is None:
			print('{0} has no grad, make sure this is good.'.format(n))


