from typing import Any
import pytorch_lightning as pl # pytorch lightning (pytorch 一个集成)
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler 
from model.resnet import Resnet34, Resnet34ExtendTorch, Resnet50ExtendTorch, Resnet18ExtendTorch

from torch import nn as nn 
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch 
from timm.models.layers import trunc_normal_
from einops import rearrange, einsum
from torch.nn import functional as F 


class Classifier(nn.Module):
    def __init__(self, in_dims, num_cls) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_dims, num_cls, 1)
    def forward(self, X):
        '''
        X : [n, 512, 1, 1]
        '''
        X = self.conv(X) # [bs n 20]
        return X 

class AdaptiveLayer(nn.Module):
    def __init__(self, in_dim, n_ratio, out_dim):
        super().__init__()
        hidden_dim = int(in_dim * n_ratio)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

'''
Wapper 
Custom hook function 
'''
class Net(pl.LightningModule):
    def __init__(self, info = "", num_classes = 20, is_cls_cam = False):
        super().__init__()
        
        # 
        self.save_hyperparameters()
        # Initial the resnet34 model
        self.model = Resnet18ExtendTorch( num_classes=num_classes)
        # Initial the loss  function 
        self.loss = nn.CrossEntropyLoss() 
        # Initial model hyparameters
        self.num_classes = num_classes # the number of classes
        self.is_cls_cam = is_cls_cam
        # self.l_fc = AdaptiveLayer(512, 4, 2048)
        # self.text_features = nn.Parameter(self._load_text_features(), requires_grad=False)
        
        # self.bce_logits_loss = nn.BCEWithLogitsLoss()
    def _load_text_features(self):
        return torch.load('/home/bai_gairui/multi_label/encoding/text/class_text_featurers.pt')
    def forward(self, X):
        X, cam_feature, cam = self.model(X)
        return X, cam_feature, cam 
    def classifier(self, X):
        '''
        X : [bs, num_class, hidden_features]
        '''
        return self.model.classifier(X)
    def classifier_with_text(self, X, target):
        target = target.float()
        X, cam_feature, cam = self(X) # cam feature : bs 20 emb , X : bs 2048 7 7 
        bs, c, w, h = X.shape 
        text_features = self.text_features.type_as(X) # bs 20 512
        text_features = self.l_fc(text_features) # bs 20 2048 
        text_features = text_features / text_features.norm(dim = -1, keepdim = True)
        
        X = rearrange(X, "b c w h -> b w h c")
        X = X / X.norm(dim = -1, keepdim = True) # bs 7 7 2048
        X = rearrange(X, "b w h c -> b (w h) c")
        logits = X @ text_features.T  # bs 49 20
        logits = rearrange(logits, "b (w h) c -> b c w h", w = w, h = h)
        
        logits = F.adaptive_avg_pool2d(logits, (1, 1)).reshape(-1, self.num_classes) # b 20
        
        loss = self.bce_logits_loss(logits, target)
        cls_pred = (torch.sigmoid(logits) > 0.5).float()
        
        
        acc = ((cls_pred == target) & (target == 1)).sum() / (target == 1).sum()
        return logits, loss , acc 
    
    def general(self, batch, batch_idx):
        # [bs, 3, 224, 224]
        files, img, target = batch 
        bs = img.shape[0]
        X, cam_feature, cam = self(img) # bs 15, 512 -> bs 15 512
        # cam = cam.reshape(bs, )
        '''
        
        '''
        mask = target > 0 # bs 20
        feature_list = [cam_feature[i][mask[i]] for i in range(bs)] #  bs n 512
        if self.is_cls_cam:
            feature_list = [cam[i][mask[i]] for i in range(bs)] 
        preds = [self.classifier(f.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1) for f in feature_list] # bs n 15
        labels = [torch.nonzero(i).squeeze(-1) for i in target ] # bs n 
        loss_ce = 0 
        acc = 0 
        num = 0
        for (pred, label) in zip(preds, labels):
            loss_ce = loss_ce + self.loss(pred, label)
            acc = acc + (pred.argmax(dim = -1) == label).sum()
            num = num + pred.shape[0]
            
        
        return preds, loss_ce / bs , acc / num 
    def general2(self, batch, batch_idx):
        img, target = batch
        bs = img.shape[0]
        X, cam_feature, cam = self(img) # bs 20, 512
        

        logits, loss , acc  = self.classifier_with_text(img, target)
        
        return loss, acc
    
        
    def training_step(self, batch, batch_idx):
        _ , loss, acc = self.general(batch, batch_idx)
        self.log("loss", loss , prog_bar=True, on_epoch=True, on_step=True)
        self.log("acc", acc , prog_bar=True, on_epoch=True, on_step=True)
        return loss 
    
    def validation_step(self, batch, batch_idx):
        _, loss, acc = self.general(batch, batch_idx)
        self.log("test_loss", loss , prog_bar=True, on_epoch=True, on_step=True)
        self.log("test_acc", acc , prog_bar=True, on_epoch=True, on_step=True)
        return loss 

    def test_step(self, batch, batch_idx):
        files, X, y = batch 
        preds, loss, acc = self.general(batch, batch_idx)
        # print(preds.shape)
        return
        
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optim =  Adam(self.parameters(), lr=1e-3)
        schedular = CosineAnnealingLR(optim, 20)
        return [optim], [schedular]
            
        
        
if __name__ == "__main__":
    # X = torch.rand(3, 512).unsqueeze(-1).unsqueeze(-1)
    # cls = Classifier(512, 20)
    # print(cls(X).shape)
    
    pass 