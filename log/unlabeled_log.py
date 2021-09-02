import torch

class UnlabeledLog:
	def __init__(self, opt):
		self.opt = opt

	def begin_pass(self, shared):
		self.shared = shared
		self.log = ['x1,x2,premise,hypothesis,entail_probability,neutral_probability,contradiction_probability']
		if (not hasattr(self.opt, 'pred_output')) or self.opt.pred_output == '':
			raise Exception('pred_output must be specified when using UnlabeledPred')
		self.pred_output = self.opt.pred_output

		self.label_intensity = [0.0 for _ in self.opt.labels]	# record label-wise avg probabilities
		self.num_ex = 0

	def record(self, pred):
		dist = pred.data.exp()
		res_map = self.shared.res_map

		for k, ex_idx in enumerate(self.shared.batch_ex_idx):
			if 'x_pair' in res_map:
				self.log.append('{0},{1},{2},{3},{4:.4f},{5:.4f},{6:.4f}'.format(res_map['x_pair'][k][0], res_map['x_pair'][k][1], res_map['sent1'][k], res_map['sent2'][k], float(dist[k][0]), float(dist[k][1]), float(dist[k][2])))
			self.num_ex += 1

		for l in range(len(self.opt.labels)):
			self.label_intensity[l] += dist[:, l].sum().item()

	def end_pass(self):
		path = self.pred_output + '.pred.txt'
		print('writing predictions to {0}'.format(path))
		with open(path, 'w') as f:
			for l in self.log:
				f.write(l + '\n')

		self.label_intensity = [p / self.num_ex for p in self.label_intensity]
		print('label-wise average probabilities:')
		print(self.label_intensity)
