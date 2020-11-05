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

class ProjTransformerEncoder(nn.Module):
	def __init__(self, opt):
		super(ProjTransformerEncoder, self).__init__()
		self.opt = opt
		
		self.transformer = AutoModel.from_pretrained(self.opt.transformer_type)

		for n in self.transformer.children():
			for p in n.parameters():
				p.skip_init = True

		self.bias_proj = nn.Parameter(torch.zeros(1, self.opt.hidden_size), requires_grad=False)
		self.bias_proj.skip_init = True

		# load rotation basises if explicitly specified
		if hasattr(self.opt, 'bias_proj') and self.opt.bias_proj != '':
			print('loading bias vector from {0}...'.format(self.opt.bias_proj))
			bias_f = h5py.File(self.opt.bias_proj, 'r')
			self.bias_proj.data = torch.from_numpy(bias_f['bias'][:]).float().view(1, -1)

			# Only need to use the file pointers once
			# 	wipe out the file pointers, to avoid reloading since params will be saved into the model during training
			self.opt.bias_proj = ''


	# carefully hacked from roberta embedding forward pass from transformers v=3.4.0
	#	double check if the flow still applies when upgrading trasnformers
	#	TODO, support other transformers
	def forward(self, p_toks, h_toks):
		if 'roberta' not in self.opt.transformer_type:
			raise Exception('rotation operation is not yet supported for transformer type', self.opt.transformer_type)

		def _embeddings_projected(input_ids, token_type_ids=None):
			# Create the position ids from the input token ids. Any padded tokens remain padded.
			position_ids = create_position_ids_from_input_ids(input_ids, self.transformer.embeddings.padding_idx).to(input_ids.device)
	
			# Copied from transformers.modeling_bert.BertEmbeddings.forward
			input_shape = input_ids.size()
	
			seq_length = input_shape[1]
	
			position_ids = self.transformer.embeddings.position_ids[:, :seq_length]
	
			if token_type_ids is None:
				token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.transformer.embeddings.position_ids.device)
	
			inputs_embeds = self.transformer.embeddings.word_embeddings(input_ids)
			position_embeddings = self.transformer.embeddings.position_embeddings(position_ids)
			token_type_embeddings = self.transformer.embeddings.token_type_embeddings(token_type_ids)

			# debiasing only the word embeddings
			batch_l, seq_l, emb_size = inputs_embeds.shape
			bias_proj = self.bias_proj.view(1, 1, emb_size).expand(batch_l, 1, emb_size)
			prod = torch.bmm(inputs_embeds, bias_proj.transpose(1,2))	# (batch_l, seq_l, 1)
			inputs_embeds_projected = inputs_embeds - prod * bias_proj
			inputs_embeds_projected = inputs_embeds_projected.view(batch_l, seq_l, emb_size)

			embeddings = inputs_embeds_projected + position_embeddings + token_type_embeddings
			embeddings = self.transformer.embeddings.LayerNorm(embeddings)
			embeddings = self.transformer.embeddings.dropout(embeddings)
			return embeddings

		input_ids = torch.cat([p_toks, h_toks[:, 1:]], dim=1)

		input_shape = input_ids.size()

		attention_mask = torch.ones_like(input_ids)
		token_type_ids = torch.zeros_like(input_ids)

		extended_attention_mask: torch.Tensor = self.transformer.get_extended_attention_mask(attention_mask, input_shape, input_ids.device)
		head_mask = self.transformer.get_head_mask(head_mask=None, num_hidden_layers=self.transformer.config.num_hidden_layers)
		
		embedding_output = _embeddings_projected(input_ids, token_type_ids)

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


