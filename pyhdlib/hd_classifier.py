#!/usr/bin/env python3

''' 
==============================================================================
Associative Memory (AM) classifier for binary Hyperdimensional (HD) Comuputing 
==============================================================================
'''
import time, sys 
import torch as t 
import numpy as np
import cloudpickle as cpckl


from hd_encode import hd_encode
from am_classifier import am_classifier


__author__ = "Michael Hersche"
__email__ = "herschmi@ethz.ch"
__date__ = "17.5.2019"

class hd_classifier(am_classifier):

	def __init__(self,D=10000,encoding='sumNgramm',device='cpu',nitem=1,ngramm = 3,name='test'):
		'''	
		
		Parameters
		----------
		D : int 
			HD dimension 
		encode: hd_encoding class
			encoding class 
		'''
		self._name = name
		try:
			self.load()
		except:
			
			use_cuda = t.cuda.is_available() 
			_device = t.device(device if use_cuda else "cpu")
	
			_encoder = hd_encode(D,encoding,_device,nitem,ngramm)

			super().__init__(D,_encoder,_device)
		
	
	def save(self):
		'''	
		save class as self.name.txt
		'''
		file = open(self._name+'.txt','wb')
		cpckl.dump(self.__dict__,file)
		file.close()

	
	
	def load(self):
		'''
		try load self._name.txt
		'''
		file = open(self._name+'.txt','rb')


		self.__dict__ = cpckl.load(file)


	def save2binary_model(self): 
		'''
		try load self._name_bin.npz
		'''
		
		_am = bin2int(self._am.cpu().type(t.LongTensor).numpy())
		_itemMemory= bin2int(self._encoder._itemMemory.cpu().type(t.LongTensor).numpy())
		
		np.save(self._name+'bin',_n_classes = self._n_classes,_am=_am,_itemMemory=_itemMemory,_encoding=self._encoding)

		return 
	

	
def bin2int(x):
	'''
	try load self._name_bin.npz
	
	Parameters
	----------
	x : numpy array size = [u,v] 
		input array binary 
	Restults
	--------
	y : numpy array uint32 size = [u, ceil(v/32)]
	'''

	u,v = x.shape

	v_out = int(np.ceil(v/32))
	y = np.zeros((u,v_out),dtype = np.uint32)


	for uidx in range(u): 
		for vidx in range(v_out):
			for bidx in range(32): # iterate through all bit index 
				if vidx*32 + bidx < v:
					y[uidx,vidx] += x[uidx,vidx*32 + bidx] << bidx

	return y




	
