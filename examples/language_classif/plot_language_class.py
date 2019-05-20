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


sys.path.append('../../pyhdlib/')
from hd_classifier import hd_classifier
from load_data import load_data 




# data loader 
dl = load_data()

# init HD classifier 
ngramm = 3
encoding = "sumNgramm"
nitem = 40
D = 1000


hd = hd_classifier(D,encoding,nitem,ngramm)


# training
hd.am_init(dl._n_labels)
data,label = dl.get_train_item() 
data_t = t.from_numpy(data).view(1,-1)
label_t= t.from_numpy(data).view(1,-1)
while (label != -1) : 

	hd.am_update(data_t)
	data,label = dl.get_train_item() 
	data_t = t.from_numpy(data).view(1,-1)
	label_t= t.from_numpy(data).view(1,-1)





