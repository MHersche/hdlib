#!/usr/bin/env python3

''' 

=========================================
Language Classification Data Loader Class
=========================================
'''

__author__ = "Michael Hersche"
__email__ = "herschmi@ethz.ch"
__date__ = "20.5.2019"


import time, sys, glob
import numpy as np

from hd_classifier import hd_classifier

data_dir = 'data/'

class Dataset:

	def __init__(self): 
	
	# training files 
	self._langLabels = {'afr', 'bul', 'ces', 'dan', 'nld', 'deu', 'eng', 'est', 'fin', 'fra', 'ell', 'hun', 'ita', 'lav', 'lit', 'pol', 'por', 'ron', 'slk', 'slv', 'spa', 'swe'}
	self._n_labels = len(self._langLabels)
	self._tr_idx = 0
	self._tr_path = data_dir + 'training_texts/'

	# testing files 
	self._testList = glob.glob(data_dir+"/testing_texts/*.txt")
	self._n_test_labels = len(self._testList)
	self._test_idx = 0

	return 


	def get_train_item(self):
		'''	
		Load next training item 
		Return 
		----------
		char_array: np array of characters 
			Text 
		label: int 
			Label of text used in training 		
		'''

		if self._tr_idx < self._n_labels-1: 
			F = open(self._tr_path+self._langLabels[_tr_idx])
			string = F.read()
			char_array = np.array(list(string))
			F.close()
			self._tr_idx +=1
		else: 
			self._tr_idx = 0
			char_array = []

		return char_array, self._tr_idx-1

	def get_test_item(self):

		F = open(self._tr_path+self._langLabels[_tr_idx])
		string = F.read()
		char_array = np.array(list(string))
		F.close()

		return char_array, self._tr_idx


























