import sys
import torch
from torch import cuda
from transformers import *
from torch import nn
from torch.autograd import Variable
from util.holder import *
from util.util import *
from util.oscar import *
from transformers.modeling_roberta import *
from preprocess.get_projection_basis import *

class ProjTransformerEncoder(nn.Module):
	def __init__(self, opt):
		super(ProjTransformerEncoder, self).__init__()
		self.opt = opt
		
		self.transformer = AutoModel.from_pretrained(self.opt.transformer_type)
		self._tokenizer = AutoTokenizer.from_pretrained(opt.transformer_type)

		for n in self.transformer.children():
			for p in n.parameters():
				p.skip_rand_init = True

		self.bias_proj = nn.Parameter(torch.zeros(1, self.opt.hidden_size), requires_grad=False)
		self.bias_proj.skip_rand_init = True	# skip random initialization

		# load projection vector if explicitly specified
		if hasattr(self.opt, 'bias_proj') and self.opt.bias_proj != '':
			print('loading bias vector from {0}...'.format(self.opt.bias_proj))
			bias_f = h5py.File(self.opt.bias_proj, 'r')
			self.bias_proj.data = torch.from_numpy(bias_f['bias'][:]).float().view(1, -1)

			# Only need to use the file pointers once
			# 	wipe out the file pointers, to avoid reloading since params will be saved into the model during training
			self.opt.bias_proj = ''

		# extract projection basis
		else:
			print('initializing gender projection basis...')
			self._update_projection_basis()


	def _update_projection_basis(self, gender_ws=['he', 'she']):
		_, gender = get_vec(self._tokenizer, self.transformer, gender_ws, verbose=False)
		dtype = self.bias_proj.data.dtype
		self.bias_proj.data = torch.from_numpy(gender).float().view(1, -1)
		self.bias_proj.data = to_device(self.bias_proj.data.to(dtype), self.opt.gpuid)


	# TODO, support other transformers
	def forward(self, p_toks, h_toks):
		if 'roberta' not in self.opt.transformer_type:
			raise Exception('Unsupported transformer type', self.opt.transformer_type)

		def _embeddings_projected(input_ids, token_type_ids=None):
			input_shape = input_ids.size()
			position_ids = torch.arange(input_shape[1], dtype=torch.long, device=input_ids.device)
			position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
	
			inputs_embeds = self.transformer.embeddings.word_embeddings(input_ids)
			position_embeddings = self.transformer.embeddings.position_embeddings(position_ids)
			token_type_embeddings = self.transformer.embeddings.token_type_embeddings(token_type_ids)

			# debiasing only the word embeddings
			batch_l, seq_l, emb_size = inputs_embeds.shape
			bias_proj = self.bias_proj.view(1, 1, emb_size).expand(batch_l, 1, emb_size)
			prod = torch.bmm(inputs_embeds, bias_proj.transpose(1,2))	# (batch_l, seq_l, 1)
			inputs_embeds_projected = inputs_embeds - prod * bias_proj

			embeddings = inputs_embeds_projected + position_embeddings + token_type_embeddings
			embeddings = self.transformer.embeddings.LayerNorm(embeddings)
			embeddings = self.transformer.embeddings.dropout(embeddings)
			return embeddings

		input_ids = torch.cat([p_toks, h_toks[:, 1:]], dim=1)

		attention_mask = torch.ones_like(input_ids)
		token_type_ids = torch.zeros_like(input_ids)

		if self.training and self.opt.bias_update_every != -1:
			if self.shared.num_update != 0 and self.shared.num_update % self.opt.bias_update_every == 0:
				self._update_projection_basis()
		
		embedding_output = _embeddings_projected(input_ids, token_type_ids)

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


