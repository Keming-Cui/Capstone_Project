
from torch import nn as nn 
import torch 
from torch.nn import functional as F 
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import os 
from PIL import Image
from torchvision import transforms as  T


class LungDataset(Dataset):
    def __init__(self, image_root, csv_root, allow_no_finding = True) -> None:
        '''
        image_root :  folder of your image 
        csv_root : label files 
        allow_no_finding : whether classified labeled image  or not 
        '''
        super().__init__() 
        self.labels = [
            'No Finding',
            'Fibrosis',
            'Effusion',
            'Atelectasis',
            'Edema',
            'Pneumonia',
            'Consolidation',
            'Infiltration',
            'Cardiomegaly',
            'Pleural_Thickening',
            'Hernia',
            'Mass',
            'Emphysema',
            'Nodule',
            'Pneumothorax']
        self.labels = self.labels if allow_no_finding else self.labels[1:]
        self.df = pd.read_csv(csv_root) # read csv file 
        self.df.set_index("Image Index", inplace=True) # set index of data frame 
        
        self.base_dir = image_root 
        self.file_names = os.listdir(image_root) # get all files of image folder 
        
        self.image_transforms = T.Compose([
            T.CenterCrop(880),
            T.Resize((224, 224)),
            T.ToTensor(), 
            T.Normalize(mean=[0.5], std = 0.5)
        ]) # numpy array to tensor 
        
        
        self.allow_no_finding = allow_no_finding
        
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, index):
        
        file_name = self.file_names[index]
        file_root = self.base_dir + file_name
        
        # processing labels 
        labels = [i.lstrip().rstrip() for i in self.df.loc[file_name, "Finding Labels"].split("|")]
        if (not self.allow_no_finding) and ('No Finding' in  labels):
            return self.__getitem__(index + 1)
            
        label_ids = torch.zeros(size=(len(self.labels), ))
        label_ids[[self.labels.index(label) for label in labels]] = 1
        label_ids = label_ids.to(torch.long)   # label to 0-1 multihot encoding 
        
        # processing image
        img = Image.open(file_root) # read image 
        img = self.image_transforms(img) # transformer image to tensor 
        
        
        return file_name, img[[0], ...], label_ids
    

    

if __name__ == "__main__":
    ld = LungDataset(
        '/home/bai_gairui/multi_label/data/Chest_x_ray/images/', 
        '/home/bai_gairui/multi_label/data/Chest_x_ray/Data_Entry_2017_v2020.csv', 
        allow_no_finding=True,
    )
    # loadr = DataLoader(ld, batch_size=64, shuffle=True)
    # file_names, X, y = next(iter(loadr))
    # print(X.shape, y.shape)
    
    loader = iter(ld)
    
    while True:
        file_name, X, y = next(loader)
        print(file_name)
        print(X.shape)
        if X.shape[0] > 2:
            break
        
        print("*" * 20)
        