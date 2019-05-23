#!/usr/bin/env python3

''' 
==============================================================================
Associative Memory (AM) classifier for binary Hyperdimensional (HD) Comuputing 
==============================================================================
'''
import time, sys 
import torch as t 
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
		"""save class as self.name.txt"""
		file = open(self._name+'.txt','wb')
		cpckl.dump(self.__dict__,file)
		file.close()

	
	
	def load(self):
		"""try load self._name.txt"""
		file = open(self._name+'.txt','rb')


		self.__dict__ = cpckl.load(file)
	

	
