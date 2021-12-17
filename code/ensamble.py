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

