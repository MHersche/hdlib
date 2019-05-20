#!/usr/bin/env python3

''' 
Associative Memory (AM) classifier for binary Hyperdimensional (HD) Comuputing 
'''
import time, sys 
import torch as t 

from hd_encode import hd_encode
from am_classifier import am_classifier


__author__ = "Michael Hersche"
__email__ = "herschmi@ethz.ch"
__date__ = "17.5.2019"

class hd_classifier(am_classifer):

	def __init__(self,D,encoding,nitem=1,ngramm = 3):
		'''	
		
		Parameters
		----------
		D : int 
			HD dimension 
		encode: hd_encoding class
			encoding class 
		'''
		_encoder = hd_encoding(D,encoding,nitem,ngramm)
		super().__init__(D,_encoder)

		