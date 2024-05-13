import torch 
import torch.nn as nn 
from torch.utils.data import Dataset
import json 
import numpy as np 
from torchvision import transforms as T 
from PIL import Image 
class Voc12Dataset(Dataset):
    def __init__(self, xml_name_roots, img_base_roots, encoder_roots):
        super().__init__()
        file_names = None
        with open(xml_name_roots, 'r', encoding="utf8") as f:
            file_names = json.loads(f.read())
        self.img_file_roots = [img_base_roots + i.split(".")[0] + ".jpg" for i in file_names]
        self.encoder = np.load(encoder_roots)
        self.encoder = torch.from_numpy(self.encoder)
        self.num_classes = self.encoder.shape[1]
        
        self.img_transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ])
    def __len__(self):
        return self.encoder.shape[0]
    
    def __getitem__(self, index):
        img = Image.open(self.img_file_roots[index])
        img = self.img_transforms(img)
        label = self.encoder[index ]
        
        return img, label 
    
    
if __name__ == "__main__":
    vd = Voc12Dataset(
        xml_name_roots="/home/bai_gairui/multi_label/encoding/train/file_names_ordeg.json", 
        img_base_roots="/home/bai_gairui/multi_label/data/VOC2012_train_val/VOC2012_train_val/JPEGImages/", 
        encoder_roots="/home/bai_gairui/multi_label/encoding/multi_hot_encoding.npy"
    )
    img, label = next(iter(vd))
    print(img.shape, label.shape)