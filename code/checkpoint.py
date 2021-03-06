   
import torch
import os
import numpy as np  

def saveCheck(carpeta,nom,databaseinfo,config,modelstate,sessioninfo):	
	if not os.path.exists(carpeta):
		os.makedirs(carpeta)
	torch.save({
				'databaseinfo': databaseinfo,
				'sessionConfig': config,
				'sessioninfo': sessioninfo,
				'modelstate': modelstate,
		
	}, carpeta + nom )
	

def loadCheck(carpeta,nom):

	checkpoint = torch.load(carpeta+nom )
	databaseinfo =checkpoint['databaseinfo']
	config= checkpoint['sessionConfig']
	sessioninfo=checkpoint['sessioninfo']
	modelstate=checkpoint['modelstate']
	
	return databaseinfo,config,sessioninfo,modelstate


class databaseInfo:
	def __init__(self,problemtype,cutId,train_idx,valid_idx,test_idx):
		self.problemType=problemtype
		self.cutId=cutId
		self.train_idx = train_idx
		self.valid_idx = valid_idx
		self.test_idx = test_idx

class sessionConfig:
	def __init__(self,modelId,lr,batch_size,paciencia,max_epoch):
		self.modelId=modelId
		self.lr=lr
		self.batch_size = batch_size
		self.paciencia = paciencia
		self.max_epochs = max_epoch

class modelState:
	def __init__(self,statedic,optdic):
		self.statedic=statedic
		self.optdic=optdic

class trainSessionInfo:
	def __init__(self):
		self.epoch=0
		self.train_accus=[]
		self.train_losses=[]
		self.valid_accus=[]
		self.train_confs=[]
		self.valid_losses=[]
		self.valid_confs=[]
		
