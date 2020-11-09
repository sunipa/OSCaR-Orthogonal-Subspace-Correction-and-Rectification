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

	# TODO, support other transformers
	def forward(self, p_toks, h_toks):
		if 'roberta' not in self.opt.transformer_type:
			raise Exception('Unsupported transformer type', self.opt.transformer_type)

		def _embeddings(input_ids, token_type_ids=None):
			input_shape = input_ids.size()
			position_ids = torch.arange(input_shape[1], dtype=torch.long, device=input_ids.device)
			position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
	
			inputs_embeds = self.transformer.embeddings.word_embeddings(input_ids)
			position_embeddings = self.transformer.embeddings.position_embeddings(position_ids)
			token_type_embeddings = self.transformer.embeddings.token_type_embeddings(token_type_ids)

			embeddings = inputs_embeds + position_embeddings + token_type_embeddings
			embeddings = self.transformer.embeddings.LayerNorm(embeddings)
			embeddings = self.transformer.embeddings.dropout(embeddings)
			return embeddings

		input_ids = torch.cat([p_toks, h_toks[:, 1:]], dim=1)

		attention_mask = torch.ones_like(input_ids)
		token_type_ids = torch.zeros_like(input_ids)
		
		embedding_output = _embeddings(input_ids, token_type_ids)

		head_mask = [None] * self.transformer.config.num_hidden_layers
		extended_attention_mask = attention_mask[:, None, None, :]
		extended_attention_mask = extended_attention_mask.to(dtype=next(self.transformer.parameters()).dtype) # fp16 compatibility
		extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
		encoder_outputs = self.transformer.encoder(
			embedding_output,
			attention_mask=extended_attention_mask,
			head_mask=head_mask
		)
		sequence_output = encoder_outputs[0]
		pooled_output = self.transformer.pooler(sequence_output) if self.transformer.pooler is not None else None

		if pooled_output is not None:
			return sequence_output + pooled_output.unsqueeze(1) * 0	# just a hacky way to handle pooled layer
		return sequence_output


	# call this before running forward func to setup context for the batch
	def begin_pass(self, shared):
		self.shared = shared

	# call this before running the next forward to reset context
	def end_pass(self):
		pass


