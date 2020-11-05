import sys
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from util.holder import *
from util.util import *
from sklearn.metrics import f1_score


# Multiclass Loss
class MulticlassLoss(torch.nn.Module):
	def __init__(self, opt):
		super(MulticlassLoss, self).__init__()
		self.opt = opt
		
		
	def _prec_rec(self, pred, gold):
		for p, g in zip(pred, gold):
			if p == g:
				self.class_correct_cnt[p] += 1
			self.class_pred_cnt[p] += 1
			self.class_gold_cnt[g] += 1

	def forward(self, pred, gold):
		log_p = pred
		batch_l = self.shared.batch_l
		assert(pred.shape == (batch_l, self.opt.num_label))

		# loss
		crit = torch.nn.NLLLoss(reduction='sum')	# for pytorch < 0.4.1, use size_average=False
		if self.opt.gpuid != -1:
			crit = crit.cuda(self.opt.gpuid)
		loss = crit(log_p, gold[:])

		# stats
		self.num_correct += np.equal(pick_label(log_p.data.cpu()), gold.cpu()).sum()
		self._prec_rec(pick_label(log_p.data.cpu()), gold.cpu())
		self.num_ex += batch_l

		# other stats
		pred = pick_label(log_p.data.cpu())
		gold = gold.cpu()
		for ex_idx, p, g in zip(self.shared.batch_ex_idx, pred, gold):
			p = int(p)
			g = int(g)
			self.all_ex_idx.append(ex_idx)
			self.all_pred.append(p)
			self.all_gold.append(g)
			# update the confusion matrix
			self.conf_mat[g][p] += 1
				

		return loss


	# return a string of stats
	def print_cur_stats(self):
		stats = 'Acc {0:.3f} '.format(float(self.num_correct) / self.num_ex)
		return stats

	# get training metric (scalar metric, extra metric)
	def get_epoch_metric(self):
		#macro_f1 = f1_score(np.asarray(self.all_gold), np.asarray(self.all_pred), average='macro')  
		#weighted_f1 = f1_score(np.asarray(self.all_gold), np.asarray(self.all_pred), average='weighted')  

		cls_prec = [match / pred if pred != 0 else 0.0 for match, pred in zip(self.class_correct_cnt, self.class_pred_cnt)]
		cls_rec = [match / gold if gold != 0 else 0.0 for match, gold in zip(self.class_correct_cnt, self.class_gold_cnt)]
		cls_f1 = [2.0 * p * r / (p + r) if (p+r) != 0 else 0.0 for p, r in zip(cls_prec, cls_rec)]

		acc = float(self.num_correct) / self.num_ex
		return acc, [acc] + cls_f1 	# and any other scalar metrics	


	def begin_pass(self, shared):
		self.shared = shared

		self.num_correct = 0
		self.num_ex = 0

		self.all_ex_idx = []
		self.all_pred = []
		self.all_gold = []

		self.conf_mat = [[0 for _ in range(self.opt.num_label)] for _ in range(self.opt.num_label)]

		self.class_pred_cnt = [0 for _ in range(self.opt.num_label)]
		self.class_gold_cnt = [0 for _ in range(self.opt.num_label)]
		self.class_correct_cnt = [0 for _ in range(self.opt.num_label)]

	def end_pass(self):
		print('Confusion matrix (row by gold, column by prediction):')
		for row in self.conf_mat:
			print(row)

