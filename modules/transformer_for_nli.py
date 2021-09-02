import sys
import torch
from torch import nn
from torch import cuda
import numpy as np
from util.holder import *
from log.unlabeled_log import *
from .optimizer import *
from .transformer_encoder import *
from .oscar_transformer_encoder import *
from .proj_transformer_encoder import *
from .transformer_we_encoder import *
from .linear_classifier import *

from packaging import version
import transformers
if version.parse(transformers.__version__) < version.parse('4.0'):
	# for transformers 3+
	from transformers.modeling_roberta import RobertaPreTrainedModel
	from transformers.configuration_roberta import RobertaConfig
else:
	# for transformers 4+
	from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
	from transformers.models.roberta.configuration_roberta import RobertaConfig

############ NOTE for tokenizer loading
# as of now, the recommended way to auto-load tokenizer is to firstly load the model:
# m = AutoModel.from_pretarined(/path/to/transformer_for_nli_model/)
# t = AutoTokenizer.from_pretrained(/path/to/transformer_for_nli_model/, config=m.encoder.transformer.config)

class TransformerForNLIConfig(PretrainedConfig):
	model_type = "transformerfornli"
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

class TransformerForNLI(PreTrainedModel):
	config_class = TransformerForNLIConfig
	base_model_prefix = "TransformerForNLI"	# doesn't do anything, just to make the model load properly

	def __init__(self, config, *model_args, **model_kwargs):
		super().__init__(config)
		
		# options can be overwritten by externally specified ones
		if 'global_opt' in model_kwargs:
			for k, v in model_kwargs['global_opt'].__dict__.items():
				setattr(config, k, v)
			for k, v in config.__dict__.items():
				setattr(model_kwargs['global_opt'], k, v)

		if config.enc == 'transformer':
			self.encoder = TransformerEncoder(config)
		elif config.enc == 'oscar_transformer':
			self.encoder = OSCaRTransformerEncoder(config)
		elif config.enc == 'proj_transformer':
			self.encoder = ProjTransformerEncoder(config)
		elif config.enc == 'transformer_we':
			self.encoder = TransformerWEEncoder(config)
		elif config.enc == 'oscar_transformer_we':
			self.encoder = OSCaRTransformerWEEncoder(config)
		elif config.enc == 'proj_transformer_we':
			self.encoder = ProjTransformerWEEncoder(config)
		else:
			raise Exception('unrecognized encoder type: ', config.enc)

		if config.cls == 'linear':
			self.classifier = LinearClassifier(config)
		else:
			raise Exception('unrecognized classifier type: ', config.cls)

		self.log = None
		if hasattr(config, 'log') and config.log != '':
			if config.log == 'unlabeled':
				self.log = UnlabeledLog(config)
			else:
				raise Exception('unrecognized log type: ', config.log)

	# Copied from transformers.modeling_bert.BertPreTrainedModel._init_weights
	def _init_weights(self, module):
		""" Initialize the weights """
		if isinstance(module, (nn.Linear, nn.Embedding)):
			# Slightly different from the TF version which uses truncated_normal for initialization
			# cf https://github.com/pytorch/pytorch/pull/5617
			module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
		elif isinstance(module, nn.LayerNorm):
			module.bias.data.zero_()
			module.weight.data.fill_(1.0)
		if isinstance(module, nn.Linear) and module.bias is not None:
			module.bias.data.zero_()

	def init_weight(self):
		missed_names = []
		if self.config.param_init_type == 'xavier_uniform':
			for n, p in self.named_parameters():
				if p.requires_grad and not hasattr(p, 'skip_rand_init'):
					if 'weight' in n:
						print('initializing {}'.format(n))
						nn.init.xavier_uniform_(p)
					elif 'bias' in n:
						print('initializing {}'.format(n))
						nn.init.constant_(p, 0)
					else:
						missed_names.append(n)
				else:
					missed_names.append(n)
		elif self.config.param_init_type == 'xavier_normal':
			for n, p in self.named_parameters():
				if p.requires_grad and not hasattr(p, 'skip_rand_init'):
					if 'weight' in n:
						print('initializing {}'.format(n))
						nn.init.xavier_normal_(p)
					elif 'bias' in n:
						print('initializing {}'.format(n))
						nn.init.constant_(p, 0)
					else:
						missed_names.append(n)
				else:
					missed_names.append(n)
		elif self.config.param_init_type == 'no':
			for n, p in self.named_parameters():
				missed_names.append(n)
		else:
			assert(False)

		if len(missed_names) != 0:
			print('uninitialized fields: {0}'.format(missed_names))

		model_parameters = filter(lambda p: p.requires_grad, self.parameters())
		num_params = sum([np.prod(p.size()) for p in model_parameters])
		print('total number of trainable parameters: {0}'.format(num_params))


	def __forward(self, p_toks, h_toks):
		# encoder
		enc = self.encoder(p_toks, h_toks)

		# classifier
		output = self.classifier(enc)

		# logging
		if self.log is not None:
			self.log.record(output)

		return output

	def forward(self, p_toks, h_toks):
		if 'amp' in self.config.optim:
			with autocast():
				return self.__forward(p_toks, h_toks)
		return self.__forward(p_toks, h_toks)


	def begin_pass(self, shared):
		self.shared = shared
		self.encoder.begin_pass(self.shared)
		self.classifier.begin_pass(self.shared)
		if self.log is not None:
			self.log.begin_pass(self.shared)

	def end_pass(self):
		self.encoder.end_pass()
		self.classifier.end_pass()
		if self.log is not None:
			self.log.end_pass()

	def distribute(self):
		modules = []
		modules.append(self.encoder)
		modules.append(self.classifier)

		for m in modules:
			if hasattr(m, 'customize_cuda_id'):
				print('pushing module to customized cuda id: {0}'.format(m.customize_cuda_id))
				m.cuda(m.customize_cuda_id)
			else:
				print('pushing module to default cuda id: {0}'.format(self.config.gpuid))
				m.cuda(self.config.gpuid)


	def get_param_dict(self):
		is_cuda = self.config.gpuid != -1
		param_dict = {}
		skipped_fields = []
		for n, p in self.named_parameters():
			# save all parameters that do not have skip_save flag
			# 	unlearnable parameters will also be saved
			if not hasattr(p, 'skip_save') or p.skip_save == 0:
				param_dict[n] =  torch2np(p.data, is_cuda)
			else:
				skipped_fields.append(n)
		#print('skipped fields:', skipped_fields)
		return param_dict

	def set_param_dict(self, param_dict):
		skipped_fields = []
		rec_fields = []
		for n, p in self.named_parameters():
			if n in param_dict:
				rec_fields.append(n)
				# load everything we have
				print('setting {0}'.format(n))
				p.data.copy_(torch.from_numpy(param_dict[n][:]))
			else:
				skipped_fields.append(n)
		print('skipped fileds: {0}'.format(skipped_fields))


# So that we can just call AutoModel.from_pretrained once TransformerForNLI is imported
if version.parse(transformers.__version__) < version.parse('4.0'):
	transformers.modeling_auto.MODEL_MAPPING[TransformerForNLIConfig] = TransformerForNLI
	transformers.configuration_auto.MODEL_NAMES_MAPPING['transformerfornli'] = 'TransformerForNLI'
	transformers.configuration_auto.CONFIG_MAPPING['transformerfornli'] = TransformerForNLIConfig
else:
	transformers.models.auto.modeling_auto.MODEL_MAPPING[TransformerForNLIConfig] = TransformerForNLI
	transformers.models.auto.configuration_auto.MODEL_NAMES_MAPPING['transformerfornli'] = 'TransformerForNLI'
	transformers.models.auto.configuration_auto.CONFIG_MAPPING['transformerfornli'] = TransformerForNLIConfig

