import torch

class UnlabeledPred:
	def __init__(self, opt):
		self.opt = opt

	def begin_pass(self, shared):
		self.shared = shared
		self.log = ['x1,x2,premise,hypothesis,entail_probability,neutral_probability,contradiction_probability']
		if (not hasattr(self.opt, 'pred_output')) or self.opt.pred_output == '':
			raise Exception('pred_output must be specified when using UnlabeledPred')
		self.pred_output = self.opt.pred_output

	def record(self, pred):
		dist = pred.data.exp()
		res_map = self.shared.res_map

		for k, ex_idx in enumerate(self.shared.batch_ex_idx):
			if 'x_pair' in res_map:
				self.log.append('{0},{1},{2},{3},{4:.4f},{5:.4f},{6:.4f}'.format(res_map['x_pair'][k][0], res_map['x_pair'][k][1], ' '.join(res_map['sent1'][k]), ' '.join(res_map['sent2'][k]), float(dist[k][0]), float(dist[k][1]), float(dist[k][2])))

	def end_pass(self):
		path = self.pred_output + '.pred.txt'
		print('writing predictions to {0}'.format(path))
		with open(path, 'w') as f:
			for l in self.log:
				f.write(l + '\n')
