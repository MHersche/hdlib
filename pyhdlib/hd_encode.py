#!/usr/bin/env python3

''' 
=================
HD encoding class
=================
'''
import torch as t 


__author__ = "Michael Hersche"
__email__ = "herschmi@ethz.ch"
__date__ = "17.5.2019"



class hd_encode():
	def __init__(self,D,encoding,device,nitem=1,ngramm = 3):
		'''	
		Encoding 
		Parameters
		----------
		encoding: string 
			Encoding architecture {"sumNgramm"}
		nitem: int
			number of items in itemmemory 
		ngramm: int
			number of ngramms

		
		'''
		self._D = D
		self._device = device
		# encoding scheme 
		if encoding =="sumNgramm":
			self.encode = self._compute_sumNgramm 
			self._ngramm = ngramm
			# malloc for Ngramm block, ngramm result, and sum vector  
			self._block = t.Tensor(self._ngramm,self._D).zero_().to(self._device)
			self._Y = t.Tensor(self._D).to(self._device)
			self._SumVec= t.Tensor(self._D).zero_().to(self._device)

		else: 
			raise ValueError("No valid encoding! got "+ code)

		# item memory initialization 
		self._itemMemory = t.randint(0,2,(nitem,D)).to(self._device)


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


	def _compute_sumNgramm(self,X,clip=False):
		'''	
		compute sum of ngramms 
		Parameters
		----------
		X: torch tensor, size = [n_samples,n_feat] 
			feature vectors
		Return
		------

		'''
		# reset block to zero
		self._block.zero_().to(self._device)
		self._SumVec.zero_()


		n_samlpes,n_feat = X.shape
		
		for feat_idx in range(n_feat): 
			self._SumVec.add_(self._ngrammencoding(X[0],feat_idx))

		if clip: 
			self._SumVec = self._threshold(self._SumVec,n_feat)
			n_feat = 1
			
		# put here clipping option 
		return self._SumVec, n_feat

	def _ngrammencoding(self,X,start):
		'''	
		Load next ngramm

		Parameters
		----------
		X: Torch tensor, size = [n_samples, D]
			Training samples 

		Results
		-------
		Y: Torch tensor, size = [D,]
		'''

		# rotate shift current block 
		for i in range(self._ngramm-1): 
			self._block[i+1] = self._circshift(self._block[i],1)
		# write new first entry 
		self._block[0] = self._lookupItemMemory(X[start])

		# calculate ngramm of _block
		self._Y = self._block[0]
		for i in range(1,self._ngramm):
			self._Y = self._bind(self._Y,self._block[i])

		return self._Y


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
		return t.cat((X[-n:], X[:-n]))

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
		# X1!= X2
		return ((t.mul((-2*X1+1), (2*X2-1))+1)/2)

	def _threshold(self,X,cnt):
		'''	
		Threshold a vector to binary 
		Parameters
		----------
		X : Torch tensor, size = [D,]
			input vector to be thresholded
		cnt: int 
			number of added binary vectors, used for determininig threshold 
			
		Results
		-------
		Y: Torch tensor, size = [D,]
			thresholded vector
		'''
		# even 
		if cnt % 2 == 0: 
			X.add_(t.randint(0,2,(self._D,)).type(t.FloatTensor).to(self._device)) # add random vector 
			cnt += 1
		
		return (X > (cnt/2)).type(t.cuda.FloatTensor)


	




		

		
