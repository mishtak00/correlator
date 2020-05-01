import numpy as np
from utils import theta



def setup_parallel_env_dscan(i_, alphas_rad_, deltas_rad_, N_C_a_d_, N_G_a_d_, theta_edges_, N_bins_theta_, theta_idx_max_, theta_min_, d_theta_):
	global i
	i = i_
	global alphas_rad
	alphas_rad = alphas_rad_
	global deltas_rad
	deltas_rad = deltas_rad_
	global N_C_a_d
	N_C_a_d = N_C_a_d_
	global N_G_a_d
	N_G_a_d = N_G_a_d_
	global theta_idx_max
	theta_idx_max = theta_idx_max_
	global theta_min
	theta_min = theta_min_
	global d_theta_rad
	d_theta_rad = d_theta_
	global theta_edges
	theta_edges = theta_edges_
	global N_bins_theta
	N_bins_theta = N_bins_theta_


def delta_scan(j: int):

	alpha_C_rad, delta_C_rad = alphas_rad[i], deltas_rad[j]
	N_C_a_d_entry = N_C_a_d[i, j]

	# penalize double counting within identical cells
	N_G_a_d[i, j] //= 2

	# multiply the current N_C alpha delta entry with a square of side 2*theta_max from the N_G field
	a_lower, a_higher = max(0, i-theta_idx_max), min(i+theta_idx_max, N_G_a_d.shape[0])
	d_lower, d_higher = max(0, j-theta_idx_max), min(j+theta_idx_max, N_G_a_d.shape[1])

	# print('a_max: {}\na_lower: {}, a_higher: {}\nd_max: {}\nd_lower: {}, d_higher: {}'
	# 	.format(N_G_a_d.shape[0], a_lower, a_higher, N_G_a_d.shape[1], d_lower, d_higher))
	
	# calculate the corresponding weights to each entry in the f_theta array
	N_C_N_G_ad_field = np.ravel(N_C_a_d_entry * N_G_a_d[a_lower : a_higher, d_lower : d_higher])
	
	# calculate the unweighted number of entries per each theta in the f_theta array
	alpha_field, delta_field = np.meshgrid(alphas_rad[a_lower:a_higher], deltas_rad[d_lower:d_higher], sparse=True)
	theta_idx_field = np.ravel(np.array((theta(alpha_C_rad, delta_C_rad, alpha_field, delta_field)-theta_min) // d_theta_rad, dtype=int).T)
	
	# print(theta_idx_field)
	# print('N_C_N_G_ad_field.shape:', N_C_N_G_ad_field.shape)
	# print('theta_idx_field.shape:', theta_idx_field.shape)
	
	theta_idx_edges = range(N_bins_theta+1)
	f_theta_entry, _ = np.histogram(theta_idx_field, bins=theta_idx_edges, weights=N_C_N_G_ad_field)

	# return the N_G_a_d[i, j] cell to its original value
	N_G_a_d[i, j] *= 2

	# if any(f_theta_entry):
	# 	print(i, j, '\n', f_theta_entry)

	return f_theta_entry


def setup_parallel_env_gscan(alphas_rad_, deltas_rad_, alpha_G_min_, delta_G_min_, r_G_min_, d_alpha_, d_delta_, d_r_, theta_idx_max_, theta_min_, d_theta_, N_C_a_d_, theta_idx_edges_):
	global alphas_rad
	alphas_rad = alphas_rad_
	global deltas_rad
	deltas_rad = deltas_rad_
	global alpha_G_min
	alpha_G_min = alpha_G_min_
	global delta_G_min
	delta_G_min = delta_G_min_
	global r_G_min
	r_G_min = r_G_min_
	global d_alpha
	d_alpha = d_alpha_
	global d_delta
	d_delta = d_delta_
	global d_r
	d_r = d_r_
	global theta_idx_max
	theta_idx_max = theta_idx_max_
	global theta_min
	theta_min = theta_min_
	global d_theta_rad
	d_theta_rad = d_theta_
	global N_C_a_d
	N_C_a_d = N_C_a_d_
	global theta_idx_edges
	theta_idx_edges = theta_idx_edges_


def galaxy_scan(alpha_G, delta_G, r_G, w_G):
	alpha_G_rad, delta_G_rad = np.deg2rad(alpha_G), np.deg2rad(delta_G)
	# Slice up the parts of N_G_a_d that we need here
	i = int((alpha_G - alpha_G_min) // d_alpha)
	j = int((delta_G - delta_G_min) // d_delta)
	# makes indices for slicing NC_ad
	a_lower, a_higher = max(0, i-theta_idx_max), min(i+theta_idx_max, N_C_a_d.shape[0])
	d_lower, d_higher = max(0, j-theta_idx_max), min(j+theta_idx_max, N_C_a_d.shape[1])
	# print('a_low: {} a_high: {} d_low: {} d_high: {}'.format(a_lower, a_higher, d_lower, d_higher))
	alpha_field, delta_field = np.meshgrid(alphas_rad[a_lower:a_higher], deltas_rad[d_lower:d_higher], sparse=True)
	# print('alpha_field: {}\ndelta_field: {}'.format(alpha_field, delta_field))

	# TODO: why does this need to be transposed?
	theta_idx_field = np.ravel(np.array((theta(alpha_G_rad, delta_G_rad, alpha_field, delta_field)-theta_min) // d_theta_rad, dtype=int).T)
	w_G_N_C_ad_field = np.ravel(w_G * N_C_a_d[a_lower : a_higher, d_lower : d_higher])
	r_idx = int((r_G - r_G_min) // d_r)

	g_theta_r_entry, _ = np.histogram(theta_idx_field, bins=theta_idx_edges, weights=w_G_N_C_ad_field)

	return g_theta_r_entry, r_idx


def setup_parallel_env_cscan(alphas_rad_, deltas_rad_, alpha_C_min_, delta_C_min_, r_C_min_, d_alpha_, d_delta_, d_r_, theta_idx_max_, theta_min_, d_theta_, N_G_a_d_, theta_idx_edges_):
	global alphas_rad
	alphas_rad = alphas_rad_
	global deltas_rad
	deltas_rad = deltas_rad_
	global alpha_C_min
	alpha_C_min = alpha_C_min_
	global delta_C_min
	delta_C_min = delta_C_min_
	global r_C_min
	r_C_min = r_C_min_
	global d_alpha
	d_alpha = d_alpha_
	global d_delta
	d_delta = d_delta_
	global d_r
	d_r = d_r_
	global theta_idx_max
	theta_idx_max = theta_idx_max_
	global theta_min
	theta_min = theta_min_
	global d_theta_rad
	d_theta_rad = d_theta_
	global N_G_a_d
	N_G_a_d = N_G_a_d_
	global theta_idx_edges
	theta_idx_edges = theta_idx_edges_


def center_scan(alpha_C, delta_C, r_C, w_C):
	alpha_C_rad, delta_C_rad = np.deg2rad(alpha_C), np.deg2rad(delta_C)
	# Slice up the parts of N_G_a_d that we need here
	i = int((alpha_C - alpha_C_min) // d_alpha)
	j = int((delta_C - delta_C_min) // d_delta)
	# print('i: {} j: {}'.format(i,j))
	a_lower, a_higher = max(0, i-theta_idx_max), min(i+theta_idx_max, N_G_a_d.shape[0])
	d_lower, d_higher = max(0, j-theta_idx_max), min(j+theta_idx_max, N_G_a_d.shape[1])
	# print('a_low: {} a_high: {} d_low: {} d_high: {}'.format(a_lower, a_higher, d_lower, d_higher))
	alpha_field, delta_field = np.meshgrid(alphas_rad[a_lower:a_higher], deltas_rad[d_lower:d_higher], sparse=True)
	# print('alpha_field: {}\ndelta_field: {}'.format(alpha_field, delta_field))

	# TODO: use iterator ndarray.flat instead of ravel. flat() guarantees an iterator on the original object, whereas ravel might copy the array 
	theta_idx_field = np.ravel(np.array((theta(alpha_C_rad, delta_C_rad, alpha_field, delta_field)-theta_min) // d_theta_rad, dtype=int).T)
	w_C_N_G_ad_field = np.ravel(w_C * N_G_a_d[a_lower : a_higher, d_lower : d_higher])
	r_idx = int((r_C - r_C_min) // d_r)

	g_theta_r_CG_entry, _ = np.histogram(theta_idx_field, bins=theta_idx_edges, weights=w_C_N_G_ad_field)

	return g_theta_r_CG_entry, r_idx






