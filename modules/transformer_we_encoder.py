import sys
import torch
from torch import cuda
from transformers import *
from torch import nn
from torch.autograd import Variable
from util.holder import *
from util.util import *

from packaging import version
import transformers
if version.parse(transformers.__version__) < version.parse('4.0'):
	from transformers.modeling_roberta import *
else:
	from transformers.models.roberta.modeling_roberta import *

# a version of transformer encoder that only uses word embeddings
class TransformerWEEncoder(nn.Module):
	def __init__(self, opt):
		super(TransformerWEEncoder, self).__init__()
		self.opt = opt
		
		self.transformer = AutoModel.from_pretrained(self.opt.transformer_type)

		for n in self.transformer.children():
			for p in n.parameters():
				p.skip_rand_init = True


	def __roberta_embeddings(self, input_ids):
		past_key_values_length = 0
		# Create the position ids from the input token ids. Any padded tokens remain padded.
		position_ids = create_position_ids_from_input_ids(input_ids, self.transformer.embeddings.padding_idx, past_key_values_length).to(input_ids.device)

		input_shape = input_ids.size()
		token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.transformer.embeddings.position_ids.device)
		token_type_embeddings = self.transformer.embeddings.token_type_embeddings(token_type_ids)

		inputs_embeds = self.transformer.embeddings.word_embeddings(input_ids)

		embeddings = inputs_embeds + token_type_embeddings * 0	# hacky way of ignoring token_type_emb without potentially breaking things
		if self.transformer.embeddings.position_embedding_type == "absolute":
			position_embeddings = self.transformer.embeddings.position_embeddings(position_ids)
			embeddings += position_embeddings
		embeddings = self.transformer.embeddings.LayerNorm(embeddings)
		embeddings = self.transformer.embeddings.dropout(embeddings)
		return embeddings



	# modified version of RobertaModel.forward
	def __roberta_forward(self, input_ids, return_dict=False):
		input_shape = input_ids.size()
		batch_size, seq_length = input_shape

		device = input_ids.device if input_ids is not None else inputs_embeds.device

		embedding_output = self.__roberta_embeddings(input_ids)

		# past_key_values_length
		past_key_values_length = 0

		attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
		#token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

		# We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
		# ourselves in which case we just need to make it broadcastable to all heads.
		extended_attention_mask: torch.Tensor = self.transformer.get_extended_attention_mask(attention_mask, input_shape, device)

		encoder_extended_attention_mask = None

		head_mask = None
		# Prepare head mask if needed
		# 1.0 in head_mask indicate we keep the head
		# attention_probs has shape bsz x n_heads x N x N
		# input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
		# and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
		head_mask = self.transformer.get_head_mask(head_mask, self.transformer.config.num_hidden_layers)

		encoder_outputs = self.transformer.encoder(
			embedding_output,
			attention_mask=extended_attention_mask,
			head_mask=head_mask,
			return_dict=False)

		sequence_output = encoder_outputs[0]
		pooled_output = self.transformer.pooler(sequence_output) if self.transformer.pooler is not None else None

		if not return_dict:
			return (sequence_output, pooled_output) + encoder_outputs[1:]

		return BaseModelOutputWithPoolingAndCrossAttentions(
			last_hidden_state=sequence_output,
			pooler_output=pooled_output,
			past_key_values=encoder_outputs.past_key_values,
			hidden_states=encoder_outputs.hidden_states,
			attentions=encoder_outputs.attentions,
			cross_attentions=encoder_outputs.cross_attentions,
		)
				

	def forward(self, p_toks, h_toks):
		input_ids = torch.cat([p_toks, h_toks[:, 1:]], dim=1)

		last, pooled = self.__roberta_forward(input_ids, return_dict=False)
		last = last + pooled.unsqueeze(1) * 0	# just a hacky way to handle pooled layer
		return last


	# call this before running forward func to setup context for the batch
	def begin_pass(self, shared):
		self.shared = shared

	# call this before running the next forward to reset context
	def end_pass(self):
		pass

