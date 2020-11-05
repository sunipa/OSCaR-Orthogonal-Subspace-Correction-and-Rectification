import io
import h5py
import torch
from torch import nn
from torch import cuda
import numpy as np
import ujson
from util.util import *
from util.holder import *

class Data():
	def __init__(self, opt, data_file, res_files=None, triple_mode=False):
		self.opt = opt
		self.data_name = data_file

		print('loading data from {0}'.format(data_file))
		f = h5py.File(data_file, 'r')
		self.source = f['source'][:]	# indices to glove tokens
		self.target = f['target'][:]	# indices to glove tokens
		self.source_l = f['source_l'][:].astype(np.int32)
		self.target_l = f['target_l'][:].astype(np.int32)
		self.label = f['label'][:]
		self.batch_l = f['batch_l'][:].astype(np.int32)
		self.batch_idx = f['batch_idx'][:].astype(np.int32)
		self.ex_idx = f['ex_idx'][:].astype(np.int32)
		self.length = self.batch_l.shape[0]

		self.source = torch.from_numpy(self.source)
		self.target = torch.from_numpy(self.target)
		self.label = torch.from_numpy(self.label)

		self.batches = []
		for i in range(self.length):
			start = self.batch_idx[i]
			end = start + self.batch_l[i]

			# get example token indices
			source_i = self.source[start:end, 0:self.source_l[i]]
			target_i = self.target[start:end, 0:self.target_l[i]]
			label_i = self.label[start:end]

			# sanity check
			assert(self.source[start:end, self.source_l[i]:].sum() == 0)
			assert(self.target[start:end, self.target_l[i]:].sum() == 0)

			self.batches.append((source_i, target_i,
				int(self.batch_l[i]), int(self.source_l[i]), int(self.target_l[i]), label_i))

		# count examples
		self.num_ex = 0
		for i in range(self.length):
			self.num_ex += self.batch_l[i]


		# load resource files
		self.res_names = []
		if res_files is not None:
			for f in res_files:
				if f.endswith('txt'):
					res_names = self.__load_txt(f)

				elif f.endswith('json'):
					res_names = self.__load_json_res(f)

				else:
					assert(False)
				self.res_names.extend(res_names)


	def subsample(self, ratio, minimal_num=0):
		target_num_ex = int(float(self.num_ex) * ratio)
		target_num_ex = max(target_num_ex, minimal_num)
		sub_idx = torch.LongTensor(range(self.size()))
		sub_num_ex = 0

		if ratio != 1.0:
			rand_idx = torch.randperm(self.size())
			i = 0
			while sub_num_ex < target_num_ex and i < self.batch_l.shape[0]:
				sub_num_ex += self.batch_l[rand_idx[i]]
				i += 1
			sub_idx = rand_idx[:i]

		else:
			sub_num_ex = self.batch_l.sum()

		return sub_idx, sub_num_ex

	def split(self, sub_idx, ratio):
		num_ex = sum([self.batch_l[i] for i in sub_idx])
		target_num_ex = int(float(num_ex) * ratio)

		cur_num_ex = 0
		cur_pos = 0
		for i in range(len(sub_idx)):
			cur_pos = i
			cur_num_ex += self.batch_l[sub_idx[i]]
			if cur_num_ex >= target_num_ex:
				break

		return sub_idx[:cur_pos+1], sub_idx[cur_pos+1:], cur_num_ex, num_ex - cur_num_ex


	def __load_txt(self, path):
		lines = []
		print('loading resource from {0}'.format(path))
		# read file in unicode mode!!!
		with io.open(path, 'r+', encoding="utf-8") as f:
			for l in f:
				lines.append(l.rstrip())
		# the second last extension is the res name
		res_name = path.split('.')[-2]
		res_data = lines[:]

		# some customized parsing
		parsed = []
		parsed = res_data

		setattr(self, res_name, parsed)
		return [res_name]


	def __load_json_res(self, path):
		print('loading resource from {0}'.format(path))
		
		f_str = None
		with open(path, 'r') as f:
			f_str = f.read()
		j_obj = ujson.loads(f_str)

		# get key name of the file
		assert(len(j_obj) == 2)
		res_type = next(iter(j_obj))

		res_name = None
		if j_obj[res_type] == 'map':
			res_name = self.__load_json_map(path)
		elif j_obj[res_type] == 'list':
			res_name = self.__load_json_list(path)
		else:
			assert(False)

		return [res_name]

	
	def __load_json_map(self, path):
		f_str = None
		with open(path, 'r') as f:
			f_str = f.read()
		j_obj = ujson.loads(f_str)

		assert(len(j_obj) == 2)

		res_name = None
		for k, v in j_obj.items():
			if k != 'type':
				res_name = k

		# optimize indices
		res = {}
		for k, v in j_obj[res_name].items():
			lut = {}
			for i, j in v.items():
				if i == res_name:
					lut[res_name] = [int(l) for l in j]
				else:
					lut[int(i)] = ([l for l in j[0]], [l for l in j[1]])

			res[int(k)] = lut
		
		setattr(self, res_name, res)
		return res_name


	def __load_json_list(self, path):
		f_str = None
		with open(path, 'r') as f:
			f_str = f.read()
		j_obj = ujson.loads(f_str)

		assert(len(j_obj) == 2)
		
		res_name = None
		for k, v in j_obj.items():
			if k != 'type':
				res_name = k

		# optimize indices
		res = {}
		for k, v in j_obj[res_name].items():
			p = v['p']
			h = v['h']

			# for token indices, shift by 1 to incorporate the nul-token at the beginning
			res[int(k)] = ([l for l in p], [l for l in h])
		
		setattr(self, res_name, res)
		return res_name


	def size(self):
		return self.length


	def __getitem__(self, idx):
		(source, target, batch_l, source_l, target_l, label) = self.batches[idx]

		# transfer to gpu if needed
		if self.opt.gpuid != -1:
			source = source.cuda(self.opt.gpuid)
			target = target.cuda(self.opt.gpuid)
			label = label.cuda(self.opt.gpuid)

		# get batch ex indices
		batch_ex_idx = [self.ex_idx[i] for i in range(self.batch_idx[idx], self.batch_idx[idx] + self.batch_l[idx])]

		res_map = self.__get_res(idx)

		batch_context = Holder()
		batch_context.source_l = source_l
		batch_context.target_l = target_l
		batch_context.batch_l = batch_l
		batch_context.data_name = self.data_name
		batch_context.res_map = res_map
		batch_context.batch_ex_idx = batch_ex_idx

		return (source, target, label), batch_context


	def __get_res(self, idx):
		# if there is no resource presents, return None
		if len(self.res_names) == 0:
			return None

		batch_ex_idx = [self.ex_idx[i] for i in range(self.batch_idx[idx], self.batch_idx[idx] + self.batch_l[idx])]

		all_res = {}
		for res_n in self.res_names:
			res = getattr(self, res_n)
			batch_res = [res[ex_id] for ex_id in batch_ex_idx]
			all_res[res_n] = batch_res

		return all_res
