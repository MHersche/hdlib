#!/usr/bin/env python3

''' 
HD encoding class
'''
import torch as t 


__author__ = "Michael Hersche"
__email__ = "herschmi@ethz.ch"
__date__ = "17.5.2019"



class hd_encode:
	def __init__(self,D,encoding,nitem=1,ngramm = 3):
		'''	
		Encoding 
		Parameters
		----------
		encoding: string 
			Encoding architecture {"ngramm"}
		nitem: int
			number of items in itemmemory 
		ngramm: int
			number of ngramms

		
		'''
		self._D = D

		# encoding scheme 
		if encoding =="ngramm":
			self._encoding = self._ngrammencoding 
			self._load_encoding = self._load_ngrammencoding
			self._ngramm = ngramm
		else: 
			raise ValueError("No valid encoding! got "+ code)

		# item memory initialization 
		self._itemMemory = t.randint(0,2,(nitem,D)).bernoulli()


		return 

	def _lookupItemMemory(self,key):
		'''	
		Encoding 
		Parameters
		----------
		key: int 
			key to itemmemory
		Return
		------
		out: Torch tensor, size=[D,]
		'''
		return self._itemMemory[key]


	def _load_ngrammencoding(self,X):
		'''	
		Prepare encoding of features
		Parameters
		----------
		X: torch tensor, size = [n_samples,n_feat] 
			feature vectors
		Return
		------
		'''
		self._X = X
		n_samlpes,n_feat = X.shape
		self._start = 0
		# malloc for Ngramm block and result
		self._block = t.Tensor(self._ngramm,self._D).zero_()
		self.Y = t.Tensor(self._D)




	def _ngrammencoding(self,X):
		'''	
		Load next ngramm

		Parameters
		----------
		X: Torch tensor, size = [n_samples, D]
			Training samples 

		Results
		-------
		Y: Torch tensor, size = [n_samples-n]
		'''

		# rotate shift current block 
		for i in range(self._ngramm-1): 
			self._block[i+1] = self._circshift(self._block[i],1)
		# write new first entry 
		self._block[0] = self._lookupItemMemory(self._X[self._start])
		self._start += 1

		# calculate ngramm of _block
		self.Y = self._block[0]
		for i in range(1,self._ngramm):
			self.Y = self._bind(self._Y,self._block[i])

		return self.Y


	def _circshift(self,X,n):
		'''	
		Load next ngramm

		Parameters
		----------
		X: Torch tensor, size = [D,]
			

		Results
		-------
		Y: Torch tensor, size = [n_samples-n]
		'''
		return torch.cat((X[-n:], X[:-n]))

	def _bind(self,X1,X2): 
		'''	
		Bind two vectors with XOR 

		Parameters
		----------
		X1: Torch tensor, size = [D,]
			input vector 1 
		X2: Torch tensor, size = [D,]
			input vector 2 
			
		Results
		-------
		Y: Torch tensor, size = [D,]
			bound vector
		'''
		return (X1 != X2)


	




		

		
