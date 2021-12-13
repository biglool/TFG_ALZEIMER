import os
import numpy as np
from tfg.code.checkpoint import saveCheck,loadCheck,databaseInfo,sessionConfig,modelState,trainSessionInfo
from tfg.code.trainingSteps import train, validate, getLoaders
from tfg.code.modelLoader import generate_model
import tfg.code.utils as ut
import torch.nn as nn
from torch import optim
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score
from sklearn.model_selection import train_test_split

def simpleTrainExperiment(carpeta, nom, device, dataset,train_idx,valid_idx,test_idx, problemType, cut, model, batch_size, lr, paciencia, verbose=False):

	# carreguem dades o inicialitzem si primer run
	if os.path.isfile(carpeta + nom ):
		databaseinfo,config,sessioninfo,modelstate= loadCheck(carpeta, nom)

		dataset.setProblem(databaseinfo.problemType)
		dataset.setCut(databaseinfo.cutId)

	else:

		dataset.setProblem(problemType)
		dataset.setCut(cut)	

		databaseinfo=databaseInfo(problemType,cut,train_idx,valid_idx,test_idx)
		config=sessionConfig(model,lr,batch_size,paciencia)		
		sessioninfo= trainSessionInfo()

	# carguem objectes 
	num_epochs = 200
	base_epoch=sessioninfo.epoch
	batch_size=config.batch_size

	model=generate_model(config.modelId)
	model.to(device)	
	optimizer = optim.Adam(model.parameters(), lr = config.lr) 	
	earlyStoper=ut.EarlyStoper(config.paciencia)
	loss_func = nn.CrossEntropyLoss() 

	loaders = getLoaders(dataset,config.batch_size,databaseinfo.train_idx,databaseinfo.valid_idx,databaseinfo.test_idx)

	if os.path.isfile(carpeta + nom ):
		model.load_state_dict(modelstate.statedic)
		optimizer.load_state_dict(modelstate.optdic)

	# run

	for epoch in range(num_epochs):
		
		train_loss,train_y_true, train_y_pred=train(model, loaders, optimizer,loss_func,batch_size, device)   
		valid_loss ,valid_y_true, valid_y_pred=validate(model, loaders, optimizer,loss_func,batch_size, device)


		train_cfm = confusion_matrix(train_y_true, train_y_pred)
		train_accu = accuracy_score(train_y_true, train_y_pred)

		valid_cfm = confusion_matrix(valid_y_true, valid_y_pred)
		valid_accu = accuracy_score(valid_y_true, valid_y_pred)
		f1=f1_score(valid_y_true, valid_y_pred)

		sessioninfo.train_accus.append(train_accu)
		sessioninfo.train_losses.append(train_loss)
		sessioninfo.valid_losses.append(valid_loss)
		sessioninfo.valid_accus.append(valid_accu)
		sessioninfo.train_confs.append(train_cfm)
		sessioninfo.valid_confs.append(valid_cfm)

		if verbose:
			ut.printEpochResult(epoch+base_epoch,train_accu,train_loss,valid_accu,valid_loss,train_cfm,valid_cfm)

		stopSave = earlyStoper.update(f1)

		if stopSave==True:
			sessioninfo.epoch= epoch+base_epoch
			modelstate= modelState(model.state_dict(),optimizer.state_dict())
			saveCheck(carpeta,nom,databaseinfo,config,modelstate,sessioninfo)   

			if verbose:
				ut.printConf(train_cfm,dataset.classes())
				ut.printConf(valid_cfm,dataset.classes())
				ut.printGrafs( sessioninfo.train_accus, sessioninfo.train_losses, sessioninfo.valid_accus, sessioninfo.valid_losses)		

		if earlyStoper.paciencia==0:
			break

def validateModel(carpeta, nom, device, dataset):

	if os.path.isfile(carpeta + nom ):
		databaseinfo,config,sessioninfo,modelstate= loadCheck(carpeta, nom)
		dataset.setProblem(databaseinfo.problemType)
		dataset.setCut(databaseinfo.cutId) 
		batch_size=config.batch_size
		model=generate_model(config.modelId)
		model.to(device)	
		optimizer = optim.Adam(model.parameters(), lr = config.lr) 	
		loss_func = nn.CrossEntropyLoss() 
		loaders = getLoaders(dataset,config.batch_size,databaseinfo.train_idx,databaseinfo.valid_idx,databaseinfo.test_idx)   
		model.load_state_dict(modelstate.statedic)
		optimizer.load_state_dict(modelstate.optdic)
		loss,y_true, y_pred=validate(model, loaders, optimizer,loss_func,batch_size, device,'test')
		print("epoch:")
		print(sessioninfo.epoch)
		metrics= ut.getMetrics(loss,y_true, y_pred)
		ut.printMetrics(metrics)

		#test_cfnorm= metrics[8] / metrics[8].astype(np.float).sum(axis=1, keepdims=True) 
		#ut.printConf(test_cfnorm,dataset.classes())
