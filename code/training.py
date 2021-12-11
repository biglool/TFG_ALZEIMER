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

def simpleTrainExperiment(carpeta, nom, device, dataset, problemType, cut, model, batch_size, lr, paciencia, verbose=False):

	# carreguem dades o inicialitzem si primer run
	if os.path.isfile(carpeta + nom ):
		databaseinfo,config,sessioninfo,modelstate= loadCheck(carpeta, nom)

		dataset.setProblem(databaseinfo.problemType)
		dataset.setCut(databaseinfo.cutId)

	else:

		dataset.setProblem(problemType)
		dataset.setCut(cut)	

	    	#fem split si no s'ha fet (70/15/15)
		train_idx, valid_idx, train_targs, valid_targs = train_test_split(np.arange(dataset.__len__()),dataset.targets(),test_size=0.3,shuffle=True, stratify=dataset.targets())
		valid_idx, test_idx= train_test_split(valid_idx,test_size=0.5,shuffle=True, stratify=valid_targs)

		databaseinfo=databaseInfo(problemType,cut,train_idx,valid_idx,test_idx)
		config=sessionConfig(model,lr,batch_size,paciencia)		
		sessioninfo= trainSessionInfo()

	# carguem objectes 
	num_epochs = 200
	batch_size=config.batch_size

	model=generate_model(config.modelId)
	model.to(device)	
	optimizer = optim.Adam(model.parameters(), lr = config.lr) 	
	earlyStoper=ut.EarlyStoper(config.paciencia)
	loss_func = nn.CrossEntropyLoss() 

	loaders = getLoaders(dataset,config.batch_size,databaseinfo.train_idx,databaseinfo.valid_idx,databaseinfo.test_idx)

	if os.path.isfile(carpeta + nom ):
		model.load_state_dict(modelState.statedic)
		optimizer.load_state_dict(modelState.optdic)

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
			ut.printEpochResult(epoch+sessioninfo.epoch,train_accu,train_loss,valid_accu,valid_loss,train_cfm,valid_cfm)

		stopSave = earlyStoper.update(f1)

		if stopSave==True:
			sessioninfo.epoch= epoch+sessioninfo.epoch
			modelstate= modelState(model.state_dict(),optimizer.state_dict())
			saveCheck(carpeta,nom,databaseinfo,config,modelstate,sessioninfo)   

			if verbose:
				ut.printConf(train_cfm)
				ut.printConf(valid_cfm)
				ut.printGrafs( sessioninfo.train_accus, sessioninfo.train_losses, sessioninfo.valid_accus, sessioninfo.valid_losses)		

		if earlyStoper.paciencia==0:
			break
