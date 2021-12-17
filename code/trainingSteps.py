import torch
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold


def train(model, loaders, optimizer,loss_func,batch_size, device):

	model.train()

	y_pred = []
	y_true = []
	running_loss = 0.0
	epoch_steps = 0


	for i, (images, labels) in enumerate(loaders['train']):
		
		images=images.to(device)
		labels=labels.to(device)

		optimizer.zero_grad()
		outputs=model(images)
		loss=loss_func(outputs,labels)
		loss.backward()
		optimizer.step()

		# conf
		preds = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
		y_pred.extend(preds) 

		labels = labels.data.cpu().numpy()
		y_true.extend(labels) 

		running_loss += loss.item()
		epoch_steps += 1

		pass

	return running_loss/epoch_steps , y_true, y_pred

def validate(model, loaders,optimizer,loss_func,batch_size, device,val_type='valid', scheduler=None):

	model.eval()

	y_pred = []
	y_true = []
	val_loss = 0.0
	val_steps = 0

	with torch.no_grad():
		for i, (images, labels) in enumerate(loaders[val_type]):
		
			images=images.to(device)
			labels=labels.to(device)

			outputs=model(images)
			loss=loss_func(outputs,labels)

			# info
			preds = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
			y_pred.extend(preds) 

			labels = labels.data.cpu().numpy()
			y_true.extend(labels) 

			val_loss += loss.cpu().numpy()
			val_steps += 1

			pass
		if scheduler is not None:
			scheduler.step(val_loss / val_steps)

	return val_loss/val_steps , y_true, y_pred

def getLoaders(dataset, batch_size,train_idx,valid_idx,test_idx, random=True):
	
	if random:
		train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
		valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)
		test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)
		
		loaders = {
		'train' : torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler),   
		'valid'  : torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler),
		'test'  : torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler),
		}
		
	elif random==False:
		
		train_dataset = torch.utils.data.Subset(dataset, train_idx)
		valid_dataset = torch.utils.data.Subset(dataset,valid_idx)
		test_dataset = torch.utils.data.Subset(dataset,test_idx)

		loaders = {
			'train' : torch.utils.data.DataLoader(train_dataset, batch_size=batch_size),   
			'valid'  : torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size),
			'test'  : torch.utils.data.DataLoader(test_dataset, batch_size=batch_size),
		}

	return loaders

def getSplits(dataset):
	#(70/15/15)
	train_idx, valid_idx, train_targs, valid_targs = train_test_split(np.arange(dataset.__len__()),dataset.targets(),test_size=0.3,shuffle=True, stratify=dataset.targets())
	valid_idx, test_idx= train_test_split(valid_idx,test_size=0.5,shuffle=True, stratify=valid_targs)
	return train_idx, valid_idx , test_idx

def getKfoldSlits(dataset, nslits=5):
	
	skf = StratifiedKFold(n_splits=nslits)
	folds=[]
	for train_idx,test_idx in skf.split(np.arange(dataset.__len__()),dataset.targets()):
  		folds.append([train_idx,test_idx])
	return folds
