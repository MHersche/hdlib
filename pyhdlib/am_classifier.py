#!/usr/bin/env python3

''' 
Associative Memory (AM) classifier for binary Hyperdimensional (HD) Comuputing 
'''
import time, sys 
import torch as t 


__author__ = "Michael Hersche"
__email__ = "herschmi@ethz.ch"
__date__ = "17.5.2019"

class am_classifier:

	def __init__(self,D,code):
		'''	
		
		Parameters
		----------
		D : int 
			HD dimension 
		encode: hd_encoding class
			encoding class 
		'''

		self._n_classes = 1
		self._D = D 

		self._code = code


	def am_init(self,n_classes):
		'''	
		Train AM 

		Parameters
		----------
		n_classes: 
		'''
		self._n_classes = n_classes
		self._am = t.Tensor(self._n_classes,self._D).zero_()
		self._cnt = t.Tensor(self._n_classes).zero_()

		return

	def am_update(self,X,y):
		'''
		Update AM 

		Parameters
		----------
		X: Torch tensor, size = [n_samples, n_feat]
			Training samples 
		y: Torch tensor, size = [n_samples]
			Training labels 
		'''

		# summation of training vectors 
		for sample in range(n_samples): 
			y_s = y[sample]
			if (y_s < self._n_classes) and (y_s >= 0):
				enc_vec, n_add = self._code.encode(X[sample])
				self._am[y_s]._add(enc_vec)
				self._cnt[y_s] += n_add
			else: 
				raise ValueError("Label is not in range of [{:},{:}], got {:}".format(0,self._n_classes,y_s))

		return

	def am_threshold(self):
		'''	
		Threshold AM 
		'''
		# Thresholding 
		for y_s in range(self._n_classes): 
			# break ties randomly by adding random vector to 
			if self._cnt[y_s] % 2 == 0: 
				self._am[y_s].add_(t.randint(0,2,(self._D,)).bernoulli()) # add random vector 
				self._cnt[y_s] += 1
			self._am[y_s] = self._am[y_s] > int(self._cnt[y_s]/2)
		return

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
		n_classes = t.max(y)+1
		self.am_init(n_classes)

		# Train am  
		self.am_update(X,y)

		# Thresholding 
		self.am_threshold()

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
			# encode samples 
			enc_vec, _ = self._code.encode(X[sample],True)
			# calculate hamming distance for every class
			for y_s in range(self._n_classes): 
				hd_dist[y_s] = 1-t.sum(X[sample] == self._am[y_s])/ float(D)

			dec_values[sample] = t.argmin(hd_dist)

		return dec_values




		

		
