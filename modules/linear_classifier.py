import sys
import torch
from torch import nn
from torch.autograd import Variable
from util.holder import *
from util.util import *

class LinearClassifier(nn.Module):
	def __init__(self, opt):
		super(LinearClassifier, self).__init__()
		self.opt = opt

		self.linear = nn.Sequential(
			nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size, opt.num_label))


	# The input enc of size (batch_l, seq_l, hidden_size)
	def forward(self, enc):
		scores = self.linear(enc[:, 0, :])	# (batch_l, num_label)
		log_p = nn.LogSoftmax(1)(scores)

		return log_p


	def begin_pass(self, shared):
		self.shared = shared

	def end_pass(self):
		pass
