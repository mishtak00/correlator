"""
Copyright (C) 2020 Gebri Mishtaku

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses.
"""

import os
import json
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import fftconvolve
from multiprocessing import Pool
import matplotlib.pyplot as plt
from utils import *
from pprocs import *
from plotcorr import *

import time

from sklearn.neighbors import KDTree



class Correlator(object):

	def __init__(self, galaxy_file: str, center_file: str = None, gd_wtd: bool = False, 
		params_file: str = None, save: bool = False, printout: bool = False):

		self.save = save
		self.filename = galaxy_file.split('.')[0]
		self.printout = printout

		# loads galaxy data arrays
		if not gd_wtd:
			self.D_G_ra, self.D_G_dec, self.D_G_redshift = load_data(galaxy_file)
			self.D_G_weights = np.ones(len(self.D_G_ra), dtype=float)
		else:
			self.D_G_ra, self.D_G_dec, self.D_G_redshift, self.D_G_weights = load_data_weighted(galaxy_file)

		# gets cosmology and other hyperparameters
		self.cosmology, self.grid_spacing = load_hyperparameters(params_file)
		# calculates lookup tables for fast conversion from r to z and vice versa
		self.LUT_radii, self.LUT_redshifts = interpolate_r_z(self.D_G_redshift.min(), self.D_G_redshift.max(), self.cosmology)

		self.D_G_radii = self.LUT_radii(self.D_G_redshift)

		# instance variables
		self.s_lower_bound = 5.
		self.s_upper_bound = 205.
		# self.s_lower_bound = 110.
		# self.s_upper_bound = 160.
		self.current_s = 110.
		self.d_s = 10.

		# loads center data arrays
		if center_file is not None:
			# cross-correlation
			if galaxy_file!=center_file:
				if self.printout:
					print('Starting cross-correlation...')
				self.D_C_ra, self.D_C_dec, self.D_C_redshift, self.D_C_weights = load_data_weighted(center_file)
				# self.D_C_ra += 180 # TODO: DEV ONLY, FIX THIS
			# autocorrelation
			else:
				if self.printout:
					print('Starting autocorrelation...')
				self.D_C_ra, self.D_C_dec, self.D_C_redshift = load_data(center_file)
				self.D_C_weights = np.ones(len(self.D_C_ra), dtype=float)

			self.D_C = np.array(sky2cartesian(self.D_C_ra, self.D_C_dec, self.D_C_redshift, self.LUT_radii)).T
			self.D_C_radii = self.LUT_radii(self.D_C_redshift)
		else:
			# TODO: only for dev phase, REMOVE after
			# runs cf on galaxy data
			self.vote_threshold = 80
			self._get_centers((self.D_G_ra, self.D_G_dec, self.D_G_redshift))

		# more instance variables
		self.bins_s = np.arange(self.s_lower_bound, self.s_upper_bound+self.d_s, self.d_s, dtype=int)
		self.N_bins_s = len(self.bins_s)
		self.s_idx_edges = range(self.N_bins_s+1)

		# TODO: should this be multiplied by 1/2?
		# define the angular bin d_theta_rad
		self.d_theta_rad = self.d_s / self.D_G_radii.max()
		self.d_theta_deg = np.rad2deg(self.d_theta_rad)

		self.theta_min = 0.
		self.theta_max = self.s_upper_bound / self.D_G_radii.min()
		self.theta_idx_max = int(self.theta_max // self.d_theta_rad)

		# the additions to the second and third arguments below 
		# are to accomodate the outer edge of the final bin
		self.N_bins_theta = int((self.theta_max - self.theta_min) // self.d_theta_rad + 1)
		self.theta_edges = np.linspace(self.theta_min, self.theta_max + self.d_theta_rad, self.N_bins_theta + 1, endpoint=True)
		self.theta_idx_edges = range(self.N_bins_theta+1)

		if self.printout:
			print(f'theta_max: {self.theta_max}\nd_theta: {self.d_theta_rad}')
			print(f'theta_idx_max: {self.theta_idx_max}\nN_bins_theta: {self.N_bins_theta}')

		self.stime = time.time()


	# TODO: write repr here


	def make_randoms(self):
		"""Constructs the probability maps over alpha&delta and r. """

		# define the angular bins d_alpha and d_theta_rad
		self.d_alpha = self.d_theta_deg
		D_G_N_bins_alpha = int((self.D_G_ra.max() - self.D_G_ra.min()) // self.d_alpha + 1)
		self.d_delta = self.d_theta_deg
		D_G_N_bins_delta = int((self.D_G_dec.max() - self.D_G_dec.min()) // self.d_delta + 1)

		# TODO: check again that this is correct
		# defines radial bin d_r and the maximum readial bin of interest for the RR, DGRC, DCRG steps
		self.d_r = self.d_s / 2.
		self.r_idx_max = int(self.s_upper_bound // self.d_r)
		self.D_G_N_bins_r = int((self.D_G_radii.max() - self.D_G_radii.min()) // self.d_r + 1)

		# TODO: check that these marginalization are okay to do as a way to project into alpha&delta and r
		# project the raw data onto the alpha-delta and r distributions
		D_G_grid, D_G_grid_bin_edges = np.histogramdd((self.D_G_ra, self.D_G_dec, self.D_G_radii), 
			bins=(D_G_N_bins_alpha, D_G_N_bins_delta, self.D_G_N_bins_r), weights = self.D_G_weights)

		# Galaxy randoms
		self.N_G_a_d = np.array([[np.sum(D_G_grid[a, d, :]) 
									for d in range(D_G_grid.shape[1])] 
								for a in range(D_G_grid.shape[0])])
		self.P_G_r = np.array([np.sum(D_G_grid[:, :, r]) 
								for r in range(D_G_grid.shape[2])]) \
							/ np.sum(self.D_G_weights)
		if self.printout:
			print('Gridded D_G shape:', D_G_grid.shape)
		del D_G_grid


		# TODO: maybe choose cell midpoints rather than the left edge values?
		# these are the alphas and deltas of each bin
		alphas = np.array(D_G_grid_bin_edges[0][:-1])
		deltas = np.array(D_G_grid_bin_edges[1][:-1])
		self.alphas_rad = np.deg2rad(alphas)
		self.deltas_rad = np.deg2rad(deltas)
		self.radii = np.array(D_G_grid_bin_edges[2][:-1])

		self.D_C_N_bins_r = int((self.D_C_radii.max() - self.D_C_radii.min()) // self.d_r + 1)


		# project the centers data onto the alpha-delta and r distributions
		# NOTE: D_C_grid shares the space and its dimensions with D_G_grid
		D_C_grid, _ = np.histogramdd((self.D_C_ra, self.D_C_dec, self.D_C_radii), 
			bins=D_G_grid_bin_edges, weights = self.D_C_weights)
		del D_G_grid_bin_edges

		# Center randoms
		self.N_C_a_d = np.array([[np.sum(D_C_grid[a, d, :]) 
									for d in range(D_C_grid.shape[1])] 
								for a in range(D_C_grid.shape[0])])
		# TODO: this should be a weighted sum
		self.P_C_r = np.array([np.sum(D_C_grid[:, :, r]) 
								for r in range(D_C_grid.shape[2])]) \
							/ np.sum(self.D_C_weights)

		if self.printout:
			print('Total votes: ', np.sum(self.D_C_weights))
			print('Gridded D_C shape:', D_C_grid.shape)

		del D_C_grid



	def make_f_theta(self, load: bool = False):

		print('Calculating f_theta...')
		stime = time.time()

		if load:
			self.f_theta = np.load('funcs_{}/f_theta_{}_ds_{}.npy'.format(self.filename, int(self.current_s), int(self.d_s)))
		else:
			self.f_theta = np.zeros((self.N_bins_theta,))

			# TODO: I can optimize here by actually parallelizing in the longest axis so that I perform the least data movement
			for i in range(self.N_C_a_d.shape[0]):
				if i%5==0:
					print("{} out of {}".format(i, self.N_C_a_d.shape[0]))
					print(self.f_theta)
				# TODO: this can be sped up if the probability maps are cut before getting sent to the children processes. do it.
				with Pool(initializer=setup_parallel_env_dscan, initargs=(i, self.alphas_rad, self.deltas_rad, self.N_C_a_d, self.N_G_a_d, 
							self.theta_edges, self.N_bins_theta, self.theta_idx_max, self.theta_min, self.d_theta_rad)) as pool:
					result = pool.map(delta_scan, range(self.N_C_a_d.shape[1]))
					for f_theta_entry in result:
						self.f_theta += f_theta_entry

			if self.save:
				np.save('funcs_{}/f_theta_{}_ds_{}.npy'.format(self.filename, int(self.current_s), int(self.d_s)), self.f_theta)

		print(f'Finished in {(time.time()-stime)/60.} minutes')
		print(self.f_theta)



	def make_gtr_DG_RC(self, load: bool = False):

		print('Calculating g(theta,r) for DG_RC...')
		stime = time.time()

		# TODO: We calculate g(theta, r) here but we should be calculating g(theta, z)
		# for it to be cosmology-independent. Should we leave this as is and change later on?
		# TODO: The current formula for sigma_GC assumes Omega_k = 0. Change this too.
		if load:
			self.g_theta_r = np.load('funcs_{}/g_theta_r_GC_{}_ds_{}.npy'.format(self.filename, int(self.current_s), int(self.d_s)))
		else:
			g_r_theta = np.zeros((self.D_G_N_bins_r, self.N_bins_theta)) # = g_theta_r.T
			alpha_G_min, delta_G_min, r_G_min = self.D_G_ra.min(), self.D_G_dec.min(), self.D_G_radii.min()

			with Pool(initializer=setup_parallel_env_gscan, initargs=(self.alphas_rad, self.deltas_rad,
				alpha_G_min, delta_G_min, r_G_min, self.d_alpha, self.d_delta, self.d_r, self.theta_idx_max,
				self.theta_min, self.d_theta_rad, self.N_C_a_d, self.theta_idx_edges)) as pool:

				results = pool.starmap(galaxy_scan, zip(self.D_G_ra, self.D_G_dec, self.D_G_radii, self.D_G_weights))
				for result in results:
					g_theta_r_entry, r_idx = result
					g_r_theta[r_idx] += g_theta_r_entry.T
			
			self.g_theta_r = g_r_theta.T
			if self.save:
				np.save('funcs_{}/g_theta_r_GC_{}_ds_{}.npy'.format(self.filename, int(self.current_s), int(self.d_s)), self.g_theta_r)
		print(self.g_theta_r)
		print(f'Finished in {(time.time()-stime)/60.} minutes')



	def make_gtr_DC_RG(self):

		print('Calculating g(theta,r) for DC_RG...')
		stime = time.time()
		
		# DC_RG
		g_r_theta_CG = np.zeros((self.D_C_N_bins_r, self.N_bins_theta)) # = g_theta_r_CG.T
		alpha_C_min, delta_C_min, r_C_min = self.D_C_ra.min(), self.D_C_dec.min(), self.D_C_radii.min()

		with Pool(initializer=setup_parallel_env_cscan, initargs=(self.alphas_rad, self.deltas_rad,
			alpha_C_min, delta_C_min, r_C_min, self.d_alpha, self.d_delta, self.d_r, self.theta_idx_max,
			self.theta_min, self.d_theta_rad, self.N_G_a_d, self.theta_idx_edges)) as pool:

			results = pool.starmap(center_scan, zip(self.D_C_ra, self.D_C_dec, self.D_C_radii, self.D_C_weights))
			for result in results:
				g_theta_r_CG_entry, r_idx = result
				g_r_theta_CG[r_idx] += g_theta_r_CG_entry.T
		
		self.g_theta_r_CG = g_r_theta_CG.T
		if self.save:
			np.save('funcs_{}/g_theta_r_CG_{}_ds_{}.npy'.format(self.filename, int(self.current_s), int(self.d_s)), self.g_theta_r_CG)
		# g_theta_r_CG = np.load('funcs/g_theta_r_CG_{}.npy'.format(int(current_s)))
		print(self.g_theta_r_CG)
		print(f'Finished in {(time.time()-stime)/60.} minutes')



	def make_RR(self):

		print('Calculating RR...')
		stime = time.time()

		
		self.RR = np.zeros((self.N_bins_s,))
		# print(r_idx_max)
		# print(P_C_r)

		for f_t_idx in range(self.f_theta.shape[0]):
			th = self.theta_edges[f_t_idx]/2

			for i in range(self.P_C_r.shape[0]):
				rmin, rmax = max(0, i-self.r_idx_max), min(i+self.r_idx_max, self.P_G_r.shape[0])
				r_C = self.radii[i]
				r_G_field = self.radii[rmin:rmax]
				# print('radii: ', radii.shape)
				# print('rmin, rmax = ', rmin, rmax)
				# print('sin(th): ', np.sin(th))
				sigma_CG = (r_C + r_G_field) * np.sin(th)
				pi_CG = np.abs(r_C - r_G_field) * np.cos(th)
				# print('sigma:', sigma_CG)
				# print('pi:', pi_CG)
				s_CG = np.sqrt(sigma_CG**2 + pi_CG**2)
				# print('s:', s_CG)
				s_idx = np.array((s_CG - self.s_lower_bound) // self.d_s, dtype=int)

				# print(s_idx)

				RR_weight_entry = self.f_theta[f_t_idx] * self.P_C_r[i] * self.P_G_r[rmin:rmax]
				RR_weight_entry = RR_weight_entry[s_idx<self.N_bins_s]
				s_idx = s_idx[s_idx<self.N_bins_s]

				RR_entry, _ = np.histogram(s_idx, bins=self.s_idx_edges, weights=RR_weight_entry)
				self.RR += RR_entry

		# TODO: delete f_theta in the integration phase wrapper, not here
		delattr(self, 'f_theta')
		if self.save:
			np.save('funcs_{}/RR_{}_ds_{}.npy'.format(self.filename, int(self.current_s), int(self.d_s)), self.RR)
		# RR = np.load('funcs/RR_{}.npy'.format(int(current_s)))
		print(self.RR)
		print(f'Finished in {(time.time()-stime)/60.} minutes')



	def make_DG_RC(self):

		print('Calculating DG_RC...')
		stime = time.time()

		self.DG_RC = np.zeros((self.N_bins_s,))

		for g_t_idx in range(self.g_theta_r.shape[0]):
			th = self.theta_edges[g_t_idx]/2

			for g_r_idx in range(self.g_theta_r.shape[1]):
				rmin, rmax = max(0, g_r_idx-self.r_idx_max), min(g_r_idx+self.r_idx_max, self.P_C_r.shape[0])
				r_C_field = self.radii[rmin:rmax]
				r_G = self.radii[g_r_idx]
				sigma_GC = (r_G + r_C_field) * np.sin(th)
				pi_GC = np.abs(r_G - r_C_field) * np.cos(th)

				s_GC = np.sqrt(sigma_GC**2 + pi_GC**2)
				s_idx = np.array((s_GC - self.s_lower_bound) // self.d_s, dtype=int)


				# TODO: this is a bit too dramatic, just do s_idx[s_idx<N_bins_s]
				intersect = np.intersect1d(s_idx, self.s_idx_edges[:-1])
				if len(intersect)>0:
					DG_RC_weight_entry = self.g_theta_r[g_t_idx, g_r_idx] * self.P_C_r[rmin:rmax]
					DG_RC_weight_entry = DG_RC_weight_entry[s_idx<self.N_bins_s]
					s_idx = s_idx[s_idx<self.N_bins_s]
					# print('s_idx:', s_idx)
					# print('DG_RC:', DG_RC_weight_entry)

					DG_RC_entry, _ = np.histogram(s_idx, bins=self.s_idx_edges, weights=DG_RC_weight_entry)
					self.DG_RC += DG_RC_entry
		
		# TODO: delete this in the integration phase wrapper, not here
		delattr(self, 'g_theta_r')
		if self.save:
			np.save('funcs_{}/DG_RC_{}_ds_{}.npy'.format(self.filename, int(self.current_s), int(self.d_s)), self.DG_RC)
		# DG_RC = np.load('funcs/DG_RC_{}.npy'.format(int(current_s)))
		print(self.DG_RC)
		print(f'Finished in {(time.time()-stime)/60.} minutes')


	def make_DC_RG(self):

		print('Calculating DC_RG...')
		stime = time.time()

		self.DC_RG = np.zeros((self.N_bins_s,))

		for g_t_idx in range(self.g_theta_r_CG.shape[0]):
			th = self.theta_edges[g_t_idx]/2

			for g_r_idx in range(self.g_theta_r_CG.shape[1]):
				rmin, rmax = max(0, g_r_idx-self.r_idx_max), min(g_r_idx+self.r_idx_max, self.P_G_r.shape[0])
				r_G_field = self.radii[rmin:rmax]
				r_C = self.radii[g_r_idx]
				sigma_CG = (r_C + r_G_field) * np.sin(th)
				pi_CG = np.abs(r_C - r_G_field) * np.cos(th)

				s_CG = np.sqrt(sigma_CG**2 + pi_CG**2)
				s_idx = np.array((s_CG - self.s_lower_bound) // self.d_s, dtype=int)


				intersect = np.intersect1d(s_idx, self.s_idx_edges[:-1])
				if len(intersect)>0:
					DC_RG_weight_entry = self.g_theta_r_CG[g_t_idx, g_r_idx] * self.P_G_r[rmin:rmax]
					DC_RG_weight_entry = DC_RG_weight_entry[s_idx<self.N_bins_s]
					s_idx = s_idx[s_idx<self.N_bins_s]
					# print('s_idx:', s_idx)
					# print('DC_RG:', DC_RG_weight_entry)

					DC_RG_entry, _ = np.histogram(s_idx, bins=self.s_idx_edges, weights=DC_RG_weight_entry)
					self.DC_RG += DC_RG_entry

		# TODO: delete this in the integration phase wrapper, not here
		delattr(self, 'g_theta_r_CG')
		if self.save:
			np.save('funcs_{}/DC_RG_{}_ds_{}.npy'.format(self.filename, int(self.current_s), int(self.d_s)), self.DC_RG)
		# DC_RG = np.load('funcs/DC_RG_{}.npy'.format(int(current_s)))
		print(self.DC_RG)
		print(f'Finished in {(time.time()-stime)/60.} minutes')




	# from scipy.spatial import KDTree

	# def make_DD(self):
	# 	'''
	# 	Counts galaxy-center pairs for galaxies and centers separated by (s-ds/2 , s+ds/2) 
	# 	for each s in the correlator. Each query below selects all nearest neighbor pairs 
	# 	separated by s+ds/2, so to get pairs separated by exactly (s-ds/2 , s+ds/2), 
	# 	it subtracts the number of galaxies returned from the previous query at s-ds/2.
	# 	Esentially, the queries are spherical, and deleting the data of a previous, 
	# 	smaller cocentric sphere from the current one achieves a shell effect while querying.
	# 	We start with a sphere of radius ds/2 less than our lower s bound to obtain the 
	# 	first spherical cutoff for the subsequent queries.
	# 	'''
	# 	# TODO: There's got to be a faster way to query shell-wise instead of
	# 	#		querying just spherically. Too inefficient rn...
	# 	# TODO: Construct kdtree over the longest set, either DG or DC.
	# 	#		This minimizes the number of queries.
		
	# 	# TODO: put D_G in a new method together with the D_C
	# 	# these serve to construct the kdtrees
	# 	D_G_xs, D_G_ys, D_G_zs = sky2cartesian(self.D_G_ra, self.D_G_dec, self.D_G_redshift, self.LUT_radii)
	# 	self.D_G = np.array([D_G_xs, D_G_ys, D_G_zs]).T

	# 	if len(self.D_C) > len(self.D_G):
	# 		longer_dataset = self.D_C
	# 		shorter_dataset = self.D_G
	# 	else:
	# 		longer_dataset = self.D_G
	# 		shorter_dataset = self.D_C

	# 	# calculates leafsize of the kdtrees based on length of datasets
	# 	# l for long, s for short
	# 	l_leafsize = int(len(longer_dataset) // 10000)
	# 	l_leafsize = l_leafsize if l_leafsize >= 10 else 10
	# 	s_leafsize = int(len(shorter_dataset) // 10000)
	# 	s_leafsize = s_leafsize if s_leafsize >= 10 else 10

	# 	print('Calculating DD...')
	# 	stime = time.time()

	# 	self.DD = np.zeros((self.N_bins_s,))
	# 	l_kdtree = KDTree(longer_dataset, leafsize=l_leafsize)
	# 	s_kdtree = KDTree(shorter_dataset, leafsize=s_leafsize)

	# 	neighbors_in_prev_sphere = s_kdtree.count_neighbors(l_kdtree, self.s_lower_bound-self.d_s/2.)
	# 	s_upper_bounds = self.bins_s+self.d_s/2
	# 	results = s_kdtree.count_neighbors(l_kdtree, s_upper_bounds)
	# 	for i, neighbors_in_curr_sphere in enumerate(results):
	# 		self.DD[i] = neighbors_in_curr_sphere - neighbors_in_prev_sphere
	# 		neighbors_in_prev_sphere = neighbors_in_curr_sphere
	# 	del l_kdtree, s_kdtree

	# 	self.DD = np.average(self.D_C_weights) * self.DD

	# 	# TODO: delete these in the integration stage wrapper, not here
	# 	delattr(self, 'D_G')
	# 	delattr(self, 'D_C')
	# 	if self.save:
	# 		np.save('funcs_{}/DD_{}_ds_{}.npy'.format(self.filename, int(self.current_s), int(self.d_s)), self.DD)
	# 	# DD = np.load('funcs/DD_{}.npy'.format(int(current_s)))
	# 	print(self.DD)
	# 	print(f'Finished in {(time.time()-stime)/60.} minutes')






	def make_DD(self):

		D_G_xs, D_G_ys, D_G_zs = sky2cartesian(self.D_G_ra, self.D_G_dec, self.D_G_redshift, self.LUT_radii)
		self.D_G = np.array([D_G_xs, D_G_ys, D_G_zs]).T

		# build tree on longer dataset
		if len(self.D_C) > len(self.D_G):
			longer_dataset, longer_wts = self.D_C, self.D_C_weights
			shorter_dataset, shorter_wts = self.D_G, self.D_G_weights
		else:
			longer_dataset, longer_wts = self.D_G, self.D_G_weights
			shorter_dataset, shorter_wts = self.D_C, self.D_C_weights

		# calculates leafsize of the kdtree
		leafsize = int(len(longer_dataset) // 10000)
		leafsize = leafsize if leafsize >= 10 else 10

		# uses default minkowski metric with p=2 (euclidian dist)
		kdtree = KDTree(longer_dataset, leaf_size=leafsize)

		# calculates smallest separation
		s_edges = np.append(self.bins_s-self.d_s/2, [self.bins_s[-1]+self.d_s/2])
		max_sep = s_edges[-1]
		print(s_edges)
		self.DD = np.zeros((self.N_bins_s,))

		print('Calculating DD...')
		stime = time.time()

		for i, p in enumerate(shorter_dataset):

			if i%1000==0:
				print(f'@ point {i} out of {len(shorter_dataset)}')

			# calculates points up to max separation value all in one pass
			idxs, dists = kdtree.query_radius(p.reshape(1, -1), max_sep, return_distance=True)

			# puts pairs in the correct separation bins given the distances from the kdtree
			# assigns correctly weighted galaxy pairs to their spherical shells
			wts = longer_wts[idxs[0]] * shorter_wts[i]
			DD_entry, _ = np.histogram(dists[0], bins=s_edges, weights=wts)
			self.DD += DD_entry

		# TODO: delete these in the integration stage wrapper, not here
		delattr(self, 'D_G')
		delattr(self, 'D_C')
		if self.save:
			np.save('funcs_{}/DD_{}_ds_{}.npy'.format(self.filename, int(self.current_s), int(self.d_s)), self.DD)
		# DD = np.load('funcs/DD_{}.npy'.format(int(current_s)))
		print(self.DD)
		print(f'Finished in {(time.time()-stime)/60.} minutes')




	def make_2pcf(self, load: bool = False):

		if load:
			self.bins_s = np.load('funcs_{}/bins_s_{}_ds_{}.npy'. format(self.filename, int(self.current_s), int(self.d_s)))
			self.xi_s = np.load('funcs_{}/xi_s_{}_ds_{}.npy'.format(self.filename, int(self.current_s), int(self.d_s)))
		else:

			self.DD = np.load('funcs_{}/DD_{}_ds_{}.npy'.format(self.filename, int(self.current_s), int(self.d_s)))
			self.DC_RG = np.load('funcs_{}/DC_RG_{}_ds_{}.npy'.format(self.filename, int(self.current_s), int(self.d_s)))
			self.DG_RC = np.load('funcs_{}/DG_RC_{}_ds_{}.npy'.format(self.filename, int(self.current_s), int(self.d_s)))
			self.RR = np.load('funcs_{}/RR_{}_ds_{}.npy'.format(self.filename, int(self.current_s), int(self.d_s)))


			# TODO: NORMALIZE!!! or not... we don't really need it here.
			# N_C_tot = np.sum(self.D_C_weights)
			# N_G_tot = np.sum(self.D_G_weights)
			# RR_norm = N_C_tot * N_G_tot
			self.xi_s = (self.DD - self.DC_RG - self.DG_RC + self.RR) / self.RR
			print(self.xi_s)

			if self.save:
				np.save('funcs_{}/bins_s_{}_ds_{}.npy'.format(self.filename, int(self.current_s), int(self.d_s)), self.bins_s)
				np.save('funcs_{}/xi_s_{}_ds_{}.npy'.format(self.filename, int(self.current_s), int(self.d_s)), self.xi_s)

		print(f'Correlation finished in {(time.time()-self.stime)/60.} minutes')



	def plot_2pcf(self, xlims: tuple = None):
		
		if xlims is None:
			xlims = (int(self.s_lower_bound-5), int(self.s_upper_bound+5))

		plot_xi(self.bins_s, self.xi_s, self.filename, self.d_s, xlims)



	def plot_distribution(self, distname: str, xlims: tuple = None, load: bool = False):

		if load:
			dist = np.load('funcs_{}/{}_{}_ds_{}.npy'.format(self.filename, dist_name, int(self.current_s), int(self.d_s)))
		else:
			dist = getattr(self, distname)

		if xlims is None:
			xlims = (self.s_lower_bound-5, self.s_upper_bound+5)

		plot_dist(self.bins_s, dist, distname, self.filename, self.d_s, xlims)


	def plot_input_data_histogram(self, dataset_name: str):
		plot_data_hist(self.D_G_ra, self.D_C_ra, self.filename)



	def _kernel(self, bao_radius: float, grid_spacing: float, additional_thickness: float = 0.) -> np.ndarray:
		""" THIS IS HERE JUST FOR DEV EASE. REMOVE AFTER COMPLETING CORRELATOR. """

		# this is the number of bins in each dimension axis
		# this calculation ensures an odd numbered gridding
		# the kernel construction has a distinct central bin on any given run
		kernel_bin_count = int(2 * np.ceil(bao_radius / grid_spacing) + 1)

		# this is the kernel inscribed radius in index units
		inscribed_r_idx_units = bao_radius / grid_spacing
		inscribed_r_idx_units_upper_bound = inscribed_r_idx_units + 0.5 + additional_thickness
		inscribed_r_idx_units_lower_bound = inscribed_r_idx_units - 0.5 - additional_thickness

		# central bin index, since the kernel is a cube this can just be one int
		kernel_center_index = int(kernel_bin_count / 2)
		kernel_center = np.array([kernel_center_index, ] * 3)

		# this is where the magic happens: each bin at a radial distance of bao_radius from the
		# kernel's center gets assigned a 1 and all other bins get a 0
		kernel_grid = np.array([[[1 if (np.linalg.norm(np.array([i, j, k]) - kernel_center) 
											>= inscribed_r_idx_units_lower_bound
										and np.linalg.norm(np.array([i, j, k]) - kernel_center)
											< inscribed_r_idx_units_upper_bound)
								  else 0
								  for k in range(kernel_bin_count)]
								 for j in range(kernel_bin_count)]
								for i in range(kernel_bin_count)])

		return kernel_grid



	def _vote(self, galaxy_data: tuple, radius: float, grid_spacing: float) -> (np.ndarray, list, np.ndarray):
		""" THIS IS HERE JUST FOR DEV EASE. REMOVE AFTER COMPLETING CORRELATOR. """

		# gets sky data and transforms them to cartesian
		ra, dec, redshift = galaxy_data
		xyzs = sky2cartesian(ra, dec, redshift, self.LUT_radii)

		# gets the 3d histogram (density_grid) and the grid bin coordintes in cartesian (grid_edges)
		galaxies_cartesian_coords = np.array(xyzs).T  # each galaxy is represented by (x, y, z)
		bin_counts_3d = np.array([np.ceil((xyzs[i].max() - xyzs[i].min()) / grid_spacing)
									for i in range(len(xyzs))], dtype=int)
		density_grid, observed_grid_edges = np.histogramdd(galaxies_cartesian_coords, bins=bin_counts_3d)

		# # subtracts the background
		# if background_subtract:
		# 	background, _ = project_and_sample(density_grid, observed_grid_edges)
		# 	density_grid -= background
		# 	density_grid[density_grid < 0.] = 0.

		# gets kernel
		kernel_grid = self._kernel(radius, grid_spacing)

		# this scans the kernel over the whole volume of the galaxy density grid
		# calculates the tensor inner product of the two at each step
		# and finally stores this value as the number of voters per that bin in the observed grid
		observed_grid = np.round(fftconvolve(density_grid, kernel_grid, mode='same'))

		return observed_grid, observed_grid_edges



	def _get_centers(self, galaxy_data: tuple):
		"""Finds the BAO centers through the centerfinder voting procedure. """
		found_centers_grid, found_centers_bin_edges = self._vote(galaxy_data, self.current_s, self.grid_spacing)
		
		# TODO: just find the centers_indices once and index D_C_weights by it too
		self.D_C_weights = found_centers_grid[found_centers_grid > self.vote_threshold]
		centers_indices = np.asarray(found_centers_grid > self.vote_threshold).nonzero()
		del found_centers_grid
		found_centers_bin_centers = np.array([(found_centers_bin_edges[i][:-1] + found_centers_bin_edges[i][1:]) / 2
												for i in range(len(found_centers_bin_edges))])
		del found_centers_bin_edges
		D_C_xyzs = np.array([found_centers_bin_centers[i][centers_indices[i]]
								for i in range(len(centers_indices))])
		self.D_C_ra, self.D_C_dec, self.D_C_redshift, self.D_C_radii = cartesian2sky(*D_C_xyzs, 
																					self.LUT_redshifts,
																					self.D_G_ra.min(),
																					self.D_G_ra.max())
		
		# serves for the kdtree step of DD construction
		self.D_C = D_C_xyzs.T
		del D_C_xyzs


			

