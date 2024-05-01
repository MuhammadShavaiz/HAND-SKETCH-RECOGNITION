"""
DATASET FILE FOR LOADING IMAGES, LABELS

dataset class returns tuple of label and PIL image 
"""
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class SketchDataSet(Dataset):
    def __init__(self,pathToDataset,transform=None,testMode=False) -> None:
        super().__init__()
        self.testMode = testMode
        self.imagesPath = os.path.join(pathToDataset,"images")
        self.labelsPath = os.path.join(pathToDataset,"labels.csv")
        if(transform):

            self.transform = transforms.Compose([
                transform,
                transforms.ToTensor()
            ])
        else:
          self.transform = transforms.Compose([
                transforms.ToTensor()
            ])            
        assert  os.path.exists(self.imagesPath) , f"no folder names 'images' found at {self.imagesPath}"
        assert  os.path.exists(self.labelsPath), f"no file named 'labels.csv found' at {self.labelsPath}"

        # loading the labels.csv file
        self.labelsFile = np.loadtxt(self.labelsPath,delimiter=",",dtype=np.int64)

        self.samplesInLabels = len(self.labelsFile)

    def __len__(self):
        return self.samplesInLabels

    def __getitem__(self, idx):
        sample = self.labelsFile[idx]            
        imgPath = os.path.join(self.imagesPath,str(sample[0])+".png")

        assert len(sample) == 2
        assert os.path.exists(imgPath), f"No image found at {imgPath}"

        
        img = Image.open(imgPath).convert('RGB')
        
        # custom transforms        
        if self.transform:
            img = self.transform(img)


        # converting images to dark background with white outlines
        img = 1 - img

        # because label will be used as indicies                
        label = sample[1] - 1

        if not self.testMode:
            return label,img
                
        # returning the fileName of image for output format during evalution time    
        return label,img, sample[0]
    




