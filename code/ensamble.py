import torch
import os
import numpy as np
from tfg.code.checkpoint import loadCheck,databaseInfo,sessionConfig,modelState,trainSessionInfo
from tfg.code.modelLoader import generate_model
from tfg.code.trainingSteps import getLoaders


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
				y_pred.extend(probabilities[0].data.cpu().numpy()) 

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

