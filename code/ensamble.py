import torch
import os
import numpy as np
from tfg.code.checkpoint import loadCheck,databaseInfo,sessionConfig,modelState,trainSessionInfo
from tfg.code.modelLoader import generate_model
from tfg.code.trainingSteps import getLoaders
import collections 
from collections import Counter
import tfg.code.utils as ut

def getPred(model, loaders, device,outType="preds",val_type='test'):

	model.eval()

	y_pred = []
	y_true = []

	with torch.no_grad():
		for i, (images, labels) in enumerate(loaders[val_type]):
		
			images=images.to(device)
			labels=labels.to(device)

			outputs=model(images)
      
			if outType=="preds":
				preds = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
				y_pred.extend(preds) 
			elif outType=="probs":
				sm = torch.nn.Softmax(dim=1)
				probabilities = sm(outputs) 
				y_pred.extend(probabilities.data.cpu().numpy()) 

			labels = labels.data.cpu().numpy()
			y_true.extend(labels) 

			pass

	return y_true, y_pred

def modelGetPreds(carpeta, nom, device, dataset,outType="preds",val_type='test', verbose=False):

	results=[]
	
	if os.path.isfile(carpeta + nom ):
		
		databaseinfo,config,sessioninfo,modelstate= loadCheck(carpeta, nom)
		
		dataset.setProblem(databaseinfo.problemType)
		dataset.setCut(databaseinfo.cutId) 
		model=generate_model(config.modelId)
		model.to(device)	
		loaders = getLoaders(dataset,config.batch_size,databaseinfo.train_idx,databaseinfo.valid_idx,databaseinfo.test_idx,random=False)   
		model.load_state_dict(modelstate.statedic)
		results=getPred(model, loaders, device,outType=outType,val_type=val_type)
		if verbose:
			print(results)
			
	return results

def stackModelsOutputs(models, device, dataset,outType="preds",val_type='test', verbose=False):
	labels=[]
	staked=[]
	for  carpeta, nom in models:
		y_true, y_pred=modelGetPreds(carpeta, nom, device, dataset,outType="preds",val_type='test', verbose=False)
		if len(labels)==0:
			labels =y_true
		elif not (labels ==y_true):
			print("Warning el ground truth no coincide")
		staked.append(y_pred)
			
	return labels, np.array(staked).T
			
def voteMax(models, device, dataset, verbose=False):

	true, pred = stackModelsOutputs(models, device, dataset , verbose=verbose)
	votemax=[Counter(predit).most_common(1)[0][0] for predit in pred]
	return true, votemax
	
def crossValidateVoteMax(models, device, dataset, K=5,verbose=False):
	resultats=[]
	for fold in range(1,5):
		models_fold= [[model, "fold"+ str(fold) +".pt"] for model in models]
		y_true, y_pred =voteMax(models_fold, device, dataset)
		metrics= ut.getMetrics(0,y_true, y_pred)
		resultats.append(resultat)
		if verbose:
			print("Resultat Fold:"+ str(fold) )	
			ut.printMetrics(metrics)		
	res_final= [sum(met)/len(met) for met in np.array(resultats).T.tolist()]
	if verbose:
		print("Resultat Final:" )	
		ut.printMetrics(res_final)
		
	return res_final
		
