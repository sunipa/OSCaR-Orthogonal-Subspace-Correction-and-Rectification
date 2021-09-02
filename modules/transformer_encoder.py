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
				p.skip_rand_init = True

		if hasattr(self.opt, 'freeze_emb') and self.opt.freeze_emb == 1:
			self.transformer.embeddings.word_embeddings.requires_grad_(False)
			self.transformer.embeddings.position_embeddings.requires_grad_(False)
			self.transformer.embeddings.token_type_embeddings.requires_grad_(False)

		self.__emb_overwriten = False


	def overwrite_emb(self, emb):
		assert(self.transformer.embeddings.word_embeddings.weight.shape == emb.shape)
		self.transformer.embeddings.word_embeddings.weight.data.copy_(emb)


	def forward(self, p_toks, h_toks):
		input_ids = torch.cat([p_toks, h_toks[:, 1:]], dim=1)

		if 'distil' in self.opt.transformer_type:
			last = self.transformer(input_ids, return_dict=False)[0]
		else:
			last, pooled = self.transformer(input_ids, return_dict=False)
			last = last + pooled.unsqueeze(1) * 0	# just a hacky way to handle pooled layer
		return last


	# call this before running forward func to setup context for the batch
	def begin_pass(self, shared):
		self.shared = shared

		if hasattr(self.opt, 'freeze_emb') and self.opt.emb_overwrite != self.opt.dir and not self.__emb_overwriten:
			print('loading embeddings to overwrite from {0}'.format(self.opt.emb_overwrite))
			tokenizer = AutoTokenizer.from_pretrained(self.opt.transformer_type)
			emb = load_emb(tokenizer.get_vocab(), self.opt.emb_overwrite)
			self.overwrite_emb(emb)
			self.__emb_overwriten = True

	# call this before running the next forward to reset context
	def end_pass(self):
		pass


