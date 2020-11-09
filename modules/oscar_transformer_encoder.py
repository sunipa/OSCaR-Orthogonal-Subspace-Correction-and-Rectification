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
from preprocess.get_rotation_basis import *

class OSCaRTransformerEncoder(nn.Module):
	def __init__(self, opt):
		super(OSCaRTransformerEncoder, self).__init__()
		self.opt = opt
		
		self.transformer = AutoModel.from_pretrained(self.opt.transformer_type)
		self._tokenizer = AutoTokenizer.from_pretrained(opt.transformer_type)

		for n in self.transformer.children():
			for p in n.parameters():
				p.skip_rand_init = True

		self.bias_v1 = nn.Parameter(torch.zeros(1, self.opt.hidden_size), requires_grad=False)
		self.bias_v2 = nn.Parameter(torch.zeros(1, self.opt.hidden_size), requires_grad=False)
		self.rot_mat = nn.Parameter(torch.zeros(self.opt.hidden_size, self.opt.hidden_size), requires_grad=False)

		self.bias_v1.skip_rand_init = True
		self.bias_v2.skip_rand_init = True
		self.rot_mat.skip_rand_init = True

		# load rotation basises if explicitly specified
		if hasattr(self.opt, 'bias_v1') and hasattr(self.opt, 'bias_v2') and self.opt.bias_v1 != '' and self.opt.bias_v2 != '':
			print('loading bias vector from {0}...'.format(self.opt.bias_v1))
			bias_f = h5py.File(self.opt.bias_v1, 'r')
			self.bias_v1.data = torch.from_numpy(bias_f['bias'][:]).float().view(1, -1)

			print('loading bias vector from {0}...'.format(self.opt.bias_v2))
			bias_f = h5py.File(self.opt.bias_v2, 'r')
			self.bias_v2.data = torch.from_numpy(bias_f['bias'][:]).float().view(1, -1)

			v1, v2 = max_span(self.bias_v1, self.bias_v2)
			self.rot_mat.data = get_gs_constrained(v1, v2)

			# Only need to use the file pointers once
			# 	wipe out the file pointers, to avoid reloading since params will be saved into the model during training
			# In case these pointers got specified during testing time, they will be overwritten adn thus loaded (as the loaded config will be overwritten)
			self.opt.bias_v1 = ''
			self.opt.bias_v2 = ''

		# extract rotation basis
		else:
			print('initializing gender and occupation rotation basises...')
			self._update_rotation_basis()

	def _update_rotation_basis(self, gender_ws=['he', 'she'], occupation_ws=['scientist','doctor','nurse','secretary','cleaner','maid','dancer','advocate','player','banker']):
		_, gender = get_basis(self._tokenizer, self.transformer, gender_ws, verbose=False)
		_, occupation = get_basis(self._tokenizer, self.transformer, occupation_ws, verbose=False)
		dtype = self.bias_v1.data.dtype
		self.bias_v1.data = torch.from_numpy(gender).float().view(1, -1)
		self.bias_v2.data = torch.from_numpy(occupation).float().view(1, -1)
		v1, v2 = max_span(self.bias_v1, self.bias_v2)
		self.rot_mat.data = get_gs_constrained(v1, v2)
		#
		self.bias_v1.data = to_device(self.bias_v1.data.to(dtype), self.opt.gpuid)
		self.bias_v2.data = to_device(self.bias_v2.data.to(dtype), self.opt.gpuid)
		self.rot_mat.data = to_device(self.rot_mat.data.to(dtype), self.opt.gpuid)


	# TODO, support other transformers
	def forward(self, p_toks, h_toks):
		if 'roberta' not in self.opt.transformer_type:
			raise Exception('Unsupported transformer type', self.opt.transformer_type)

		def _embeddings_corrected(input_ids, token_type_ids):
			input_shape = input_ids.size()
			position_ids = torch.arange(input_shape[1], dtype=torch.long, device=input_ids.device)
			position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
	
			inputs_embeds = self.transformer.embeddings.word_embeddings(input_ids)
			position_embeddings = self.transformer.embeddings.position_embeddings(position_ids)
			token_type_embeddings = self.transformer.embeddings.token_type_embeddings(token_type_ids)

			# debiasing only the word embeddings
			batch_l, seq_l, emb_size = inputs_embeds.shape
			inputs_embeds = inputs_embeds.view(-1, emb_size)
			inputs_embeds_corrected = correction(self.rot_mat, self.bias_v1, self.bias_v2, inputs_embeds)
			inputs_embeds_corrected = inputs_embeds_corrected.view(batch_l, seq_l, emb_size)
			#words_embeddings_corrected = words_embeddings

			embeddings = inputs_embeds_corrected + position_embeddings + token_type_embeddings
			embeddings = self.transformer.embeddings.LayerNorm(embeddings)
			embeddings = self.transformer.embeddings.dropout(embeddings)
			return embeddings

		input_ids = torch.cat([p_toks, h_toks[:, 1:]], dim=1)

		attention_mask = torch.ones_like(input_ids)
		token_type_ids = torch.zeros_like(input_ids)

		if self.training and self.opt.bias_update_every != -1:
			if self.shared.num_update != 0 and self.shared.num_update % self.opt.bias_update_every == 0:
				self._update_rotation_basis()
		
		embedding_output = _embeddings_corrected(input_ids, token_type_ids)

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
		self._update_rotation_basis()

	# call this before running the next forward to reset context
	def end_pass(self):
		pass


