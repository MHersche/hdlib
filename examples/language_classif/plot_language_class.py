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
from hd_classifier import hd_classifier, bin2int
from load_data import load_data 



training = True
testing = True
# data loader 
dl = load_data()

# init HD classifier 
ngramm = 3
encoding = "sumNgramm"
nitem = 256
D = 10000
device = 'cuda:0'


name = 'data/models/'+str(ngramm)+'gramm'


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
		
	
		print("train class {:} ".format(dl._langLabels[np.asscalar(label)]))
		# train am 
		hd.am_update(data,label)
	
	hd.am_threshold()
	hd.save()


########################## testing ########################################

if testing: 
	err = 0
	n_test = 0
	y = 0
	
	while (True): 
		# get data and push to gpu 
		X,y = dl.get_test_item()
		
		if y == -1: 
			break;
		
		y_hat = hd.predict(X)
	
		curr_err = np.sum(y_hat != y)
		n_test += len(y_hat)
		err += curr_err
		if curr_err != 0:
			print("Error: True class: {:}, Estimation: {:}".format(y,y_hat))
			
	print("Accuracy: {:}".format(1-err/n_test))
		

############################## Saving Model to binary #######################
		
	
hd.save2binary_model()
	



	
	 

	
	









