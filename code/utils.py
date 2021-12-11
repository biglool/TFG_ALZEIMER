import ipywidgets as ipyw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,ConfusionMatrixDisplay
import seaborn as sn
import sklearn.metrics as mets

class ImageSliceViewer3D:
    
    def __init__(self, volume, cmap='gray'):
        self.volume = volume
        self.cmap = cmap
        self.v = [np.min(volume), np.max(volume)]
        ipyw.interact(self.views)
    
    def views(self):

        maxZ1 = self.volume.shape[0] - 1
        maxZ2 = self.volume.shape[1] - 1
        maxZ3 = self.volume.shape[2] - 1
        ipyw.interact(self.plot_slice, 
            z1=ipyw.IntSlider(min=0, max=maxZ1, step=1, continuous_update=False, 
            description='Axial:',value=maxZ1//2), 
            z2=ipyw.IntSlider(min=0, max=maxZ2, step=1, continuous_update=False, 
            description='Coronal:',value=maxZ2//2),
            z3=ipyw.IntSlider(min=0, max=maxZ3, step=1, continuous_update=False, 
            description='Sagittal:',value=maxZ3//2))

    def plot_slice(self, z1, z2, z3):
        f,ax = plt.subplots(1,3)
        ax[0].imshow(self.volume[z1,:,:], cmap=plt.get_cmap(self.cmap), 
            vmin=self.v[0], vmax=self.v[1])
        ax[1].imshow(self.volume[:,z2,:], cmap=plt.get_cmap(self.cmap), 
            vmin=self.v[0], vmax=self.v[1])
        ax[2].imshow(self.volume[:,:,z3], cmap=plt.get_cmap(self.cmap), 
            vmin=self.v[0], vmax=self.v[1])
        plt.show()


def printEpochResult(epoch,train_accu,train_loss,valid_accu,valid_loss, train_cfm,valid_cfm):

  train_cfnorm=train_cfm / train_cfm.astype(np.float).sum(axis=1, keepdims=True)
  valid_cfnorm= valid_cfm / valid_cfm.astype(np.float).sum(axis=1, keepdims=True) 
  #https://www.baeldung.com/cs/learning-curve-ml
  print('Epoch:', epoch)
  print('Train Loss: %.3f | Train Accuracy: %.3f'%(train_loss,train_accu))
  print('Valid Loss: %.3f | Valid Accuracy : %.3f'%(valid_loss,valid_accu))
  print('Train Acc Class:', train_cfnorm.diagonal())
  print('Valid Acc Class:', valid_cfnorm.diagonal())

def printConf(cf_matrix, classes):
    ax= plt.subplot()
    df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes], columns = [i for i in classes])
    sn.heatmap(df_cm, annot=True,fmt='g', ax=ax)
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    plt.show()

def printGrafs(train_accu,train_losses,valid_accu,valid_losses):
    plt.plot(np.array(train_accu)*100)
    plt.plot(np.array(valid_accu)*100)
    plt.ylim(0, 100)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train','Valid'])
    plt.title('Train vs Valid Accuracy') 
    plt.show()

    plt.plot(np.array(train_losses))
    plt.plot(np.array(valid_losses))
    #plt.ylim(0, 3)
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend(['Train','Valid'])
    plt.title('Train vs Valid Losses')
    
    plt.show()


def getMetrics(loss,true, pred):
  metrics =[]
  metrics.append(loss)
  metrics.append(mets.accuracy_score(true, pred))
  metrics.append(mets.balanced_accuracy_score(true, pred))
  metrics.append(mets.recall_score(true, pred)) #spe
  metrics.append(mets.recall_score(true, pred, pos_label=0)) #sen
  metrics.append(mets.precision_score(true, pred))
  metrics.append(mets.f1_score(true, pred))
  metrics.append(mets.roc_auc_score(true, pred))
  metrics.append(mets.confusion_matrix(true, pred))

  return metrics

def printMetrics(metrics):
    print('Loss: %.3f | Acc: %.3f | BAcc: %.3f | SEN: %.3f | SPE: %.3f | PRE: %.3f | F1: %.3f | AUC: %.3f'% (metrics[0],metrics[1],metrics[2],metrics[3],metrics[4],metrics[5],metrics[6],metrics[7]))
	
class EarlyStoper:
    
    def __init__(self, paciencia):
        self.max_paciencia=paciencia
        self.paciencia=paciencia
        self.min_metric_detected = 0
            
    def update(self,metric):

        if self.paciencia< self.max_paciencia:
            self.paciencia -=1

        if self.min_metric_detected <metric:
            self.paciencia= self.max_paciencia-1
			
            if self.min_metric_detected <metric:
                self.min_metric_detected = metric
            return True
        return False
