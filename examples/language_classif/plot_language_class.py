#!/usr/bin/env python3

''' 

=================================================
Language Classification Using Ngramm-Sum Encoding
=================================================
'''

__author__ = "Michael Hersche"
__email__ = "herschmi@ethz.ch"
__date__ = "20.5.2019"


import time, sys 
import torch as t 
import numpy as np


sys.path.append('../../pyhdlib/')
from hd_classifier import hd_classifier
from load_data import load_data 



training = True
# data loader 
dl = load_data()

# init HD classifier 
ngramm = 3
encoding = "sumNgramm"
nitem = 256
D = 5000
device = 'cuda:0'


name = 'data/models/3gramm'


hd = hd_classifier(D,encoding,device,nitem,ngramm,name=name)



########################## training ########################################

if training: 
	hd.am_init(dl._n_labels)
	label = 0
	
	while (label != -1) : 
		# load data 
		data,label = dl.get_train_item() 
	
		if label == -1: 
			break 
		
		data_t = t.from_numpy(data).view(1,-1).type(t.LongTensor).to(hd._device)
		label_t= t.from_numpy(label).view(-1).type(t.LongTensor).to(hd._device)
		print("train class {:} ".format(dl._langLabels[np.asscalar(label)]))
		# train am 
		hd.am_update(data_t,label_t)
	
	hd.am_threshold()
	hd.save()


########################## testing ########################################
err = 0


while (False): 
	# get data and push to gpu 
	X,y = dl.get_test_item()
	X_t = t.from_numpy(X).view(1,-1).type(t.LongTensor).to(hd._device)
	y_t= t.from_numpy(y).view(-1).type(t.FloatTensor).to(hd._device)
	# estimation 

	y_hat = hd.predict(X_t)
	
	curr_err = t.sum(y_hat != y_t)
	err += curr_err
	if curr_err != 0:
		print("Error: True class: {:}, Estimation: {:}".format(y_t,y_hat))
	
	



	
	 

	
	









