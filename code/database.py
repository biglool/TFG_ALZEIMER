import nibabel as nib
import numpy as np
import pandas as pd
import torch
import os as os
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight

class ADNI_Dataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.baseInfo=pd.read_csv(os.path.join(self.img_dir, annotations_file))
        self.info = self.baseInfo
        self.transform = transform
        self.target_transform = target_transform
        self.targets_inf= self.info['Research Group']
        self.classes_inf=np.unique(self.info['Research Group'])
        self.cut_id =[2,3,2]

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir +'DATA_INORM/', str(self.info.iloc[idx, 7])+'.nii')
        image = nib.load(img_path) 
        image = image.get_fdata()
        image=self.doCut(image)
        #image = np.nan_to_num(image)
        image=np.where(image >= 2, 2,image)
        image=(image - np.nanmin(image)) / (np.nanmax(image) - np.nanmin(image))
        image = torch.from_numpy(image).float()
        image = torch.unsqueeze(image, 0)     
        label = torch.as_tensor(self.classes_inf.index(self.info.iloc[idx, 2]))
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
      
    def setProblem(self, problem):
        if problem == '3CLASS' :
            self.info = self.baseInfo.replace(['LMCI','EMCI'], 'MCI')
        if problem == 'CNvsAD' :
            self.info = self.baseInfo[self.baseInfo['Research Group'].isin(['AD', 'CN'])]
        if problem == 'CNvsMCI' :
            self.info = self.baseInfo.replace(['LMCI','EMCI'], 'MCI')
            self.info = self.info[self.info['Research Group'].isin(['MCI', 'CN'])]
        if problem == 'MCIvsAD' :
            self.info = self.baseInfo.replace(['LMCI','EMCI'], 'MCI')
            self.info = self.info[self.info['Research Group'].isin(['MCI', 'AD'])]
        if problem == 'EMCIvsLMCI' :
            self.info = self.baseInfo[self.baseInfo['Research Group'].isin(['EMCI', 'LMCI'])] 
			
        classes_inf=[]
        classes=np.unique(self.info['Research Group'])
        if 'AD' in classes:
            classes_inf.append('AD')
        if 'MCI' in classes:
            classes_inf.append('MCI')
        if 'LMCI' in classes:
            classes_inf.append('LMCI')
        if 'EMCI' in classes:
            classes_inf.append('EMCI')
        if 'CN' in classes:
            classes_inf.append('CN')

        self.targets_inf= self.info['Research Group']
        self.classes_inf=classes_inf 


    def setCut(self, cut_id):
        self.cut_id = cut_id

    def doCut(self,image):
        #[3,5,3]
        axial=image.shape[0]//3
        coronal=image.shape[1]//5
        sagital=image.shape[2]//3
        axialIni=axial*(self.cut_id[0]-1)
        axialFi=axial*self.cut_id[0]
        coronalIni=coronal*(self.cut_id[1]-1)
        coronalFi=coronal*self.cut_id[1]
        sagitalIni=sagital*(self.cut_id[2]-1)
        sagitalFi=sagital*self.cut_id[2]
        image = image[axialIni:axialFi, coronalIni:coronalFi , sagitalIni:sagitalFi]
        return image
		
    def targets(self):
        return self.targets_inf
		
    def classes(self):
        return self.classes_inf
		
    def class_weights(self):
        return compute_class_weight('balanced', classes=self.classes(), y=self.targets() )
