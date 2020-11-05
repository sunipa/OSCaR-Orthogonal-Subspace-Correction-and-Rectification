import torch
import numpy as np

# v: 1, dim
# mat: k, dim
def max_span(v, mat):
	k, dim = mat.shape
	vals = torch.abs((v * mat).sum(-1) /  v.norm() / mat.norm(dim=-1))	# (k,)
	max_idx = int(vals.argmax().data)
	return v / v.norm(), (mat[max_idx] / mat[max_idx].norm()).unsqueeze(0)


# mat: 2, dim
# return 1, dim
def basis(mat):
	v1, v2 = mat[0], mat[1]
	v2_prime = v2 - v1 * (v1 * v2).sum()
	return v2_prime / v2_prime.norm()


# v1: 1, dim
# v2: 1, dim
def get_gs_constrained(v1, v2):
	# v1: (k, dim)
	# v2: (k, dim)
	# return (k, dim)
	def proj(v1, v2):
		k, dim = v1.shape
		return (v1 * v2).sum(-1).view(k, 1) * v1 / (v1 * v1).sum(-1).view(k, 1)

	dim = v1.shape[1]
	eye = torch.eye(dim).to(v1)
	u = torch.zeros(dim, dim).to(v1)
	u[0] = v1 / v1.norm()
	v2_proj = v2 - proj(u[0:1], v2)
	u[1] = v2_proj / v2_proj.norm()

	for i in range(0, dim-2):
		p = proj(u[:i+2], eye[i:i+1].expand(i+2, dim)).sum(0)	# (dim)
		u[i+2] = eye[i] - p
		u[i+2] = u[i+2] / u[i+2].norm()
		
	return u


def masked_set(mat, mask, val):
	mask = mask.to(val)
	return mat * (1.0 - mask) + mask * val


# v1: 1, dim
# v2: 1, dim
# x: k, dim
def rotation(v1, v2, x):
	k, dim = x.shape
	v2p = basis(torch.cat([v1, v2], dim=0)).unsqueeze(0)	# (1, dim)
	x_tail = x[:, 2:]	# (k, dim-2)
	x_v1 = x.mm(v1.transpose(1,0))	# (k,1)
	x_v2p = x.mm(v2p.transpose(1,0))	# (k,1)
	v2_v1 = v2.mm(v1.transpose(1,0))	# (1,1)
	v2_v1_sqrt = (1.0 - v2_v1 * v2_v1).sqrt()	# (1,1)

	# redefining v1 and v2 and x
	v1 = torch.Tensor((1, 0)).view(1, 2).to(x)	# (1,2)
	v2 = torch.cat([v2_v1, v2_v1_sqrt], dim=-1)	# (1,2)
	x = torch.cat([x_v1, x_v2p], dim=-1)	# (k, 2)

	# get angles
	theta = torch.abs(v1.mm(v2.transpose(1,0)).acos()).view(1)	# constant
	theta_p = np.pi/2 - theta
	x_norm = x / x.norm(dim=-1).unsqueeze(-1)	# (k, 2)
	d = x_norm[:, 1]	# (k,)

	# acos by default causes trouble in fp16
	eps=1e-3
	phi = x_norm.mm(v1.transpose(1,0)).clamp(min=-1+eps,max=1-eps).acos().squeeze(-1)	# (k,)

	case1 = (phi < theta_p) * (d >= 0)	# (k,)
	case2 = (phi > theta_p) * (d >= 0)	# (k,)
	case3 = (phi >= np.pi - theta_p) * (d < 0)	# (k,)
	case4 = (phi < np.pi - theta_p) * (d < 0)	# (k,)

	theta_x = torch.zeros(phi.shape).to(phi)
	theta_x = masked_set(theta_x, case1, (phi / theta_p) * theta)
	theta_x = masked_set(theta_x, case2, (np.pi - phi) / (np.pi - theta_p) * theta)
	theta_x = masked_set(theta_x, case3, (np.pi - phi) / theta_p)
	theta_x = masked_set(theta_x, case4, phi / (np.pi - theta_p) * theta)

	R = torch.zeros(k,2,2).to(phi)
	R[:, 0,0] = theta_x.cos()
	R[:, 0,1] = -theta_x.sin()
	R[:, 1,0] = theta_x.sin()
	R[:, 1,1] = theta_x.cos()

	rotated_head = R.bmm(x.unsqueeze(-1)).squeeze(-1)	# (k, 2)
	rs = torch.cat([rotated_head, x_tail], dim=-1)	# (k, dim)
	return rs
	

def correction(mat, v1, v2, x):
	rotated = rotation(v1, v2, x.mm(mat.transpose(1, 0)))
	return rotated.mm(mat)

