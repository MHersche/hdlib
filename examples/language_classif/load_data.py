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


data_dir = 'data/'

class load_data:

	def __init__(self): 
	
		# training files 
		self._langLabels = {0:'afr', 1:'bul', 2:'ces', 3:'dan', 4:'nld', 
		5:'deu', 6:'eng', 7:'est', 8:'fin', 9:'fra', 10:'ell', 11:'hun', 
		12:'ita', 13:'lav', 14:'lit', 15:'pol', 16:'por', 17:'ron', 
		18:'slk', 19:'slv', 20:'spa', 21:'swe'}
		_test_langLabels = {0:'af', 1:'bg', 2:'cs', 3:'da', 4:'nl', 
			5:'de', 6:'en', 7:'et', 8:'fi', 9:'fr', 10:'el', 11:'hu', 
			12:'it', 13:'lv', 14:'lt', 15:'pl', 16:'pt', 17:'ro', 
			18:'sk', 19:'sl', 20:'es', 21:'sv'}
		self._test_langLabels = dict([(value, key) for key, value in _test_langLabels.items()]) 

		self._n_labels = len(self._langLabels)
		self._tr_idx = 0
		self._tr_path = data_dir + 'training_texts/'

		# testing files 
		self._testList = glob.glob(data_dir+"testing_texts/*.txt")
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
			fname=self._tr_path+self._langLabels[self._tr_idx]+'.txt'
			F = open(fname)
			string = F.read()
			char_array = np.array(list(string),'c')
			F.close()
			self._tr_idx +=1
		else: 
			self._tr_idx = 0
			char_array = np.array((100,))
		return char_array.view(np.uint8).reshape(1,-1), np.array(self._tr_idx-1).reshape(1,-1)

	def get_test_item(self):
		if self._test_idx < self._n_test_labels:
			fname = self._testList[self._test_idx]
			last_slash = fname.rfind('/')
			label =self._test_langLabels[fname[last_slash+1:last_slash+3]]
			 
			F = open(fname)
			string = F.read()
			char_array = np.array(list(string),'c')
			F.close()
			self._test_idx +=1

		else:
			char_array = np.array([])
			label = -1
			


		return char_array.view(np.uint8).reshape(1,-1), np.array(label).reshape(1,-1)


























