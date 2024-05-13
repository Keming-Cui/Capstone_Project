from torch import nn as nn 
import torch 
from torch.nn import functional as F 
# import torch 

from torchvision.models import resnet34, resnet50, resnet18

def gap2d(x, keepdims=False):
    out = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
    if keepdims:
        out = out.view(out.size(0), out.size(1), 1, 1)

    return out

class ConvAndNorm(nn.Module):
    def __init__(self, in_channle, out_channle, stride, filters, paddings = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channle, out_channels=out_channle, padding=paddings, stride=stride, kernel_size=filters)
        self.bn = nn.BatchNorm2d(out_channle)
        self.relu = nn.ReLU()
    def forward(self, X):
        X =  self.conv(X)
        X = self.bn(X)
        return self.relu(X)
    
class StageBlock(nn.Module):
    def __init__(self, in_channles, out_channles, stride, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channles, out_channels=out_channles, kernel_size=kernel_size,  padding=kernel_size // 2, stride=stride)
        self.conv2 = nn.Conv2d(in_channels=out_channles, out_channels=out_channles, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
        self.is_down_sample = in_channles == out_channles
    def forward(self, X):
        identity = X 
        X =  F.relu(self.conv1(X))
        X =  F.relu(self.conv2(X))
        if self.is_down_sample:
            return X 
        return identity + X 
    
class StaeModule(nn.Module):
    def __init__(self, depth, in_channles, out_channles):
        super().__init__()

        model_list = [
            StageBlock(in_channles=in_channles if i == 0 else out_channles,  out_channles=out_channles, stride=2 if i == 0 else 1, kernel_size=3) for i in range(depth)
        ]
        self.seq = nn.Sequential(*model_list)
    def forward(self, X):
        # X = self.block1(X)
        # X = self.block2(X)
        # X = self.block3(X)
        X = self.seq(X)
        return X 
    
        
    
        

class Resnet34(nn.Module):
    def __init__(self, in_channles, num_classes = 20):
        ## stage 0
        super().__init__()
        
        self.stage0 = nn.Sequential(
            nn.Conv2d(in_channels = in_channles, out_channels = 64, stride = 2, kernel_size= 7, padding = 3), 
            nn.BatchNorm2d(64), # normalization 
            nn.ReLU(), # 
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 
        ) # 56
        
        self.stage1 = StaeModule(3, in_channles=64, out_channles=64) # 28
        self.stage2 = StaeModule(4, in_channles=64, out_channles=128) # 28
        self.stage3 = StaeModule(6, in_channles=128, out_channles=256) # 14*14  256
        self.stage4 = StaeModule(3, in_channles=256, out_channles=512) # 7*7  512
        
        
        self.avg_pooling = nn.AvgPool2d(kernel_size=7)

        self.classifier = nn.Conv2d(512, num_classes, kernel_size=1)
        self.num_classes = num_classes
        self.out_features = 512
    def forward(self, X):
        '''
        X : [b, 3, 224, 224]
        '''
        bs = X.shape[0] # batch size
        X = self.stage0(X) # b 64 56 56 
        X = self.stage1(X) # b 64 56 56
        X = self.stage2(X) # b 128 28 28
        X = self.stage3(X) # b 256 14 14
        feature = self.stage4(X) # b 512 7 7 
        
        X = gap2d(feature, keepdims=True) # bs 512 7 7
        X = self.classifier(X) # bs 20 7 7 
        X = X.view(-1) # bs 20 * 7 * 7
        
        cam = F.conv2d(feature, self.classifier.weight) # bs 20 7 7 
        cam = F.relu(cam)
        cam = cam / (F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5) # bs 20 7 7
        cam_feature = cam.unsqueeze(2) * feature.unsqueeze(1) # bs 20 512 7 7 
        cam_feature = cam_feature.view(bs, self.num_classes, self.out_features, -1) # bs 20 512 49
        cam_feature = cam_feature.mean(dim = -1) # bs 20 512
        return feature, cam_feature, cam

class Resnet34ExtendTorch(nn.Module):
    def __init__(self, num_classes = 20, **kwargs):
        '''
        init the model parameters
        '''
        super().__init__()
        # Conv 
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1), 
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        res = resnet34() # pytorch defination
        self.dim = 512
        self.model = nn.Sequential(
            *[v for (n, v) in [*res.named_children()][:-2]]
        )
        self.classifier = nn.Conv2d(self.dim, num_classes, kernel_size=1)
        self.num_classes = num_classes
        
    def forward(self, X):
        bs = X.shape[0]
        
        X = self.conv1(X)
        '''
        Get the output of resnet `F`
        '''
        feature = self.model(X) # bs 512 7 7
        
        
        '''
        Mutiply the output of resnet and the weight of classifier, 
        '''
        cam = F.conv2d(feature, self.classifier.weight) # bs 15 7 7
        '''
        Activation Function - Relu
        '''
        cam = F.relu(cam)
        '''
        Min-max scalar 
        '''
        cam = cam / (F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5) # bs 15 7 7
        '''
        Multiply cam features and `F`
        '''
        cam_feature = cam.unsqueeze(2) * feature.unsqueeze(1) # bs 15 512 7 7 
        '''
        Get the mean of every feature map 
        '''
        cam_feature = cam_feature.view(bs, self.num_classes, self.dim, -1) # bs 15 512 49
        cam_feature = cam_feature.mean(dim = -1) # bs 15 512
        return feature, cam_feature, cam
    def activations_hook(self, grad):
        self.gradients = grad
        
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.model(x)
    
class Resnet50ExtendTorch(nn.Module):
    def __init__(self, num_classes = 20, **kwargs):
        '''
        init the model parameters
        '''
        super().__init__()
        # Conv 
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1), 
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        res = resnet50() # pytorch defination
        self.dim = 2048
        self.model = nn.Sequential(
            *[v for (n, v) in [*res.named_children()][:-2]]
        )
        self.classifier = nn.Conv2d(self.dim, num_classes, kernel_size=1)
        self.num_classes = num_classes
    
    def forward(self, X):
        bs = X.shape[0]
        
        X = self.conv1(X)
        '''
        Get the output of resnet `F`
        '''
        feature = self.model(X) # bs 2048 7 7
        cam = F.conv2d(feature, self.classifier.weight) # bs 15 7 7
        cam = F.relu(cam)
        '''
        Min-max scalar 
        '''
        cam = cam / (F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5) # bs 15 7 7
        '''
        Multiply cam features and `F`
        '''
        cam_feature = cam.unsqueeze(2) * feature.unsqueeze(1) # bs 15 512 7 7 
        '''
        Get the mean of every feature map 
        '''
        cam_feature = cam_feature.view(bs, self.num_classes, self.dim, -1) # bs 15 512 49
        cam_feature = cam_feature.mean(dim = -1) # bs 15 512
        return feature, cam_feature, cam


class Resnet18ExtendTorch(nn.Module):
    def __init__(self, num_classes = 15, **kwargs):
        '''
        init the model parameters
        '''
        super().__init__()
        # Conv 
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1), 
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        res = resnet18() # pytorch defination
        self.dim = 512
        self.model = nn.Sequential(
            *[v for (n, v) in [*res.named_children()][:-2]]
        )
        self.classifier = nn.Conv2d(self.dim, num_classes, kernel_size=1)
        self.num_classes = num_classes
    
    def forward(self, X):
        bs = X.shape[0]
        
        X = self.conv1(X)
        '''
        Get the output of resnet `F`
        '''
        feature = self.model(X) # bs 2048 7 7
        cam = F.conv2d(feature, self.classifier.weight) # bs 15 7 7
        cam = F.relu(cam)
        '''
        Min-max scalar 
        '''
        cam = cam / (F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5) # bs 15 7 7
        '''
        Multiply cam features and `F`
        '''
        cam_feature = cam.unsqueeze(2) * feature.unsqueeze(1) # bs 15 512 7 7 
        '''
        Get the mean of every feature map 
        '''
        cam_feature = cam_feature.view(bs, self.num_classes, self.dim, -1) # bs 15 512 49
        cam_feature = cam_feature.mean(dim = -1) # bs 15 512
        return feature, cam_feature, cam



if __name__ == "__main__":
    res = Resnet18ExtendTorch(in_channles=1, num_classes=15)
    X = torch.rand(64, 1, 224, 224)
    cam = res(X)
    print([i.shape for i in cam])
    