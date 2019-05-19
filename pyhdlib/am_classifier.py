#!/usr/bin/env python3

''' 
Associative Memory (AM) classifier for binary Hyperdimensional (HD) Comuputing 
'''
import time, sys 
import numpy as np
import torch as t 


__author__ = "Michael Hersche"
__email__ = "herschmi@ethz.ch"
__date__ = "17.5.2019"

class am_classifier:

	def __init__(self,D):
		'''	
		
		Parameters
		----------
		D : int 
			HD dimension 
		'''

		self._n_classes = 1

	def fit(self,X,y):
		'''	
		Train AM 

		Parameters
		----------
		X: Torch tensor, size = [n_samples, n_feat]
			Training samples 
		y: Torch tensor, size = [n_samples]
			Training labels 
		'''
		n_samples,_ = X.shape
		self._n_classes = t.max(y)+1

		self._am = t.Tensor(self._n_classes,D).zero_()
		cnt = t.Tensor(self._n_classes).zero_()

		# summation of training vectors 
		for sample in range(n_samples): 
			y_s = y[sample]
			if (y_s < self._n_classes) and (y_s >= 0):
				self._am[y_s] = am_[y_s] + X[sample]
				cnt[y_s] += 1
			else: 
				raise ValueError("Label is not in range of [{:},{:}], got {:}".format(0,self._n_classes,y_s))

		# Thresholding 
		for y_s in range(self._n_classes): 

			# break ties randomly by adding random vector to 
			if cnt[y_s] % 2 == 0: 
				self._am[y_s].add_(t.Tensor(D).bernoulli_()) # add random vector 
				cnt[y_s] += 1

			self._am[y_s] = self._am[y_s] > int(cnt[y_s]/2)

		return


	def predict(self,X):
		'''	
		Prediction 

		Parameters
		----------
		X: torch tensor, size = [n_samples, _D]
			Input samples to predict. 

		Returns
		-------
		dec_values : torch tensor, size = [n_sampels]
        	predicted values.

		'''

		n_samples, D = X.shape
		dec_values = t.Tensor(n_samples)
		hd_dist = t.Tensor(n_samples,self._n_classes)

		for sample in range(n_samples): 
			# calculate hamming distance for every class
			for y_s in range(self._n_classes): 
				hd_dist[y_s] = 1-t.sum(X[sample] == self._am[y_s])/ float(D)

			dec_values[sample] = t.argmin(hd_dist)

		return dec_values




		

		
