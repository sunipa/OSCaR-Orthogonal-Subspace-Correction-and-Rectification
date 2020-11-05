import sys
import torch
from torch import cuda
from transformers import *
from torch import nn
from torch.autograd import Variable
from util.holder import *
from util.util import *

class TransformerEncoder(nn.Module):
	def __init__(self, opt):
		super(TransformerEncoder, self).__init__()
		self.opt = opt
		
		self.transformer = AutoModel.from_pretrained(self.opt.transformer_type)

		for n in self.transformer.children():
			for p in n.parameters():
				p.skip_init = True


	# The input p_toks and h_toks should have a batch of examples of the same sequence length, i.e. of size (batch_l, seq_l) without the need for attention masks
	#	TODO, add support for seq_idx
	def forward(self, p_toks, h_toks):
		tok_idx = torch.cat([p_toks, h_toks[:, 1:]], dim=1)
		last, pooled = self.transformer(tok_idx)

		last = last + pooled.unsqueeze(1) * 0	# just a hacky way to handle pooled layer
		
		return last


	# call this before running forward func to setup context for the batch
	def begin_pass(self, shared):
		self.shared = shared

	# call this before running the next forward to reset context
	def end_pass(self):
		pass


