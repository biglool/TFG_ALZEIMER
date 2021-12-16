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

		
		
def createExperiment(carpeta,nom,problemType,cut,train_idx,test_idx,model,lr,batch_size,valid_idx=[],paciencia=0,max_epoch=200):

	databaseinfo=databaseInfo(problemType,cut,train_idx,valid_idx,test_idx)
	config=sessionConfig(model,lr,batch_size,paciencia,max_epoch)		
	sessioninfo= trainSessionInfo()

	model=generate_model(config.modelId)
	optimizer = optim.Adam(model.parameters(), lr = config.lr) 
	modelstate= modelState(model.state_dict(),optimizer.state_dict())
	
	saveCheck(carpeta,nom,databaseinfo,config,modelstate,sessioninfo)   

	
	
def trainExperiment(carpeta, nom, device, dataset, earlyStop=False, metrica ="loss",verbose=False):

	
	if os.path.isfile(carpeta + nom ):
		
		#carrega
		databaseinfo,config,sessioninfo,modelstate= loadCheck(carpeta, nom)

		dataset.setProblem(databaseinfo.problemType)
		dataset.setCut(databaseinfo.cutId)

		num_epochs = config.max_epochs
		base_epoch=sessioninfo.epoch
		batch_size=config.batch_size

		model=generate_model(config.modelId)
		model.to(device)	
		optimizer = optim.Adam(model.parameters(), lr = config.lr) 
		loss_func = nn.CrossEntropyLoss() 
		model.load_state_dict(modelstate.statedic)
		optimizer.load_state_dict(modelstate.optdic)
		
		loaders = getLoaders(dataset,config.batch_size,databaseinfo.train_idx,databaseinfo.valid_idx,databaseinfo.test_idx)
		
		if earlyStop == True:
			if metrica =="loss":
				earlyStoper=ut.EarlyStoper(config.paciencia,"min")
			elif metrica =="f1":
				earlyStoper=ut.EarlyStoper(config.paciencia,"max")
			
		# run
		for epoch in range(num_epochs+1):

			
			train_loss,train_y_true, train_y_pred=train(model, loaders, optimizer,loss_func,batch_size, device)   
			
			train_cfm = confusion_matrix(train_y_true, train_y_pred)
			train_accu = accuracy_score(train_y_true, train_y_pred)			
			sessioninfo.train_accus.append(train_accu)
			sessioninfo.train_confs.append(train_cfm)			
			sessioninfo.train_losses.append(train_loss)
			
			if len(databaseinfo.valid_idx) >0:
				
				valid_loss ,valid_y_true, valid_y_pred=validate(model, loaders, optimizer,loss_func,batch_size, device)

				valid_cfm = confusion_matrix(valid_y_true, valid_y_pred)
				valid_accu = accuracy_score(valid_y_true, valid_y_pred)
				sessioninfo.valid_losses.append(valid_loss)
				sessioninfo.valid_accus.append(valid_accu)
				sessioninfo.valid_confs.append(valid_cfm)

			if verbose:
				ut.printEpochResult(epoch+base_epoch,train_accu,train_loss,valid_accu,valid_loss,train_cfm,valid_cfm)
			
			#hem de guardar?
			saveTime = False
			
			if earlyStop == True:			
				if metrica =="loss":
					earlyStoper=ut.EarlyStoper(valid_loss)
				elif metrica =="f1":
					saveTime = earlyStoper.update(f1_score(valid_y_true, valid_y_pred))
			

			if (epoch+base_epoch) == num_epochs:
				saveTime = True

			# guardem	
			if saveTime==True:

				sessioninfo.epoch= epoch+base_epoch
				modelstate= modelState(model.state_dict(),optimizer.state_dict())
				saveCheck(carpeta,nom,databaseinfo,config,modelstate,sessioninfo)   

				if verbose:
					ut.printConf(train_cfm,dataset.classes())
					ut.printConf(valid_cfm,dataset.classes())
					ut.printGrafs( sessioninfo.train_accus, sessioninfo.train_losses, sessioninfo.valid_accus, sessioninfo.valid_losses)		
			
			# sortim
			
			if(epoch+base_epoch) == num_epochs or (earlyStop == True and earlyStoper.paciencia==0):
				break

				
def validateModel(carpeta, nom, device, dataset, verbose=False):

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
		metrics= ut.getMetrics(loss,y_true, y_pred)
		
		if verbose:		
			print("epoch:")
			print(sessioninfo.epoch)		
			ut.printMetrics(metrics)
			#test_cfnorm= metrics[8] / metrics[8].astype(np.float).sum(axis=1, keepdims=True) 
			#ut.printConf(test_cfnorm,dataset.classes())	
		
		return  [sessioninfo.epoch] + metrics
	
def crossValidate(carpeta, device, dataset, K=5, verbose=True):
	
	resultats =[]
	for i in range(1, K+1):
		nom = "fold"+ str(i) +".pt"
		
		if verbose:	
			 print("Fold:"+ str(i))
				
		resultat = validateModel(carpeta, nom, device, dataset, verbose=verbose)
		resultat.pop(0) # epoch
		resultats.append(resultat)
	
	metrics= [sum(met)/len(met) for met in np.array(resultats).T.tolist()]
	if verbose:	
		print("Resultat final:")	
		ut.printMetrics(metrics)	
	return metrics
		
	
