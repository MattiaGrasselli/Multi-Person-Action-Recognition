import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import cv2
import torchvision.transforms as transforms

class VGG16(nn.Module):
    def __init__(self,pretrained=True,input_channels=3,num_classes=1000) -> None:
        super().__init__()
        self.input_channels=input_channels
        self.num_classes=num_classes

        #If we want the pre-trained model, that gets initialized with the latest pre-trained ImageNet weights
        if pretrained:
            self.pretrained_weights=models.VGG16_BN_Weights.DEFAULT
            self.features=models.vgg16_bn(weights=self.pretrained_weights).features
        else:
            self.pretrained_weights=None
            self.features=self.features=models.vgg16_bn().features

        #If the number of channels are different, i'll modify the first convolutional layer (which it will need to be re-trained)
        if input_channels != 3:
            self.features[0]=nn.Conv2d(input_channels,out_channels=64,kernel_size=(3,3),stride=(1,1),padding=(1,1))

        self.AdaptiveAvgPool=nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc=nn.Linear(in_features=512,out_features=6,bias=True)
    
    def forward(self, x):
        #Remember that not all the layers need to be re-trained from scratch
        out = self.features(x)

        #print(out.shape)
        out = self.AdaptiveAvgPool(out)

        #print(out.shape)
        out = out.view(out.size(0), -1)
        
        #print(out.shape)
        out=F.dropout(out,p=0.5,training=self.training)
        
        out = self.fc(out)
        return out


def cross_modality_initialization(input_channels, conv1_weights):
    """
        Here we perform the cross modality initialization as stated in the paper 
        "Temporal Segment Networks for Action Recognition in Videos" from Wang et al.
    """
    mean_weights=conv1_weights.mean(axis=1)

    adapted_conv1_weights=torch.FloatTensor(64,input_channels,3,3)

    for i in range(input_channels):
        adapted_conv1_weights[:,i,:,:]=mean_weights
    
    return adapted_conv1_weights


def motion_weight_adapt(model):
    """
        Here we adapt the weights of the first convolution as stated in the paper
        "Temporal Segment Networks for Action Recognition in Videos" from Wang et al.
    """
    #Since models.VGG16_Weights.DEFAULT doesn't give me the weights directly, I download them from the url 
    #(so that if the weights changes I will download them from the latest source) 
    vgg16_pretrained_weights=model_zoo.load_url(model.pretrained_weights.value.url)
    
    #Since KeyError is raised if the keys of state_dict doesn not match the ones returned by the module's 
    #state_dict() function, I remove all the keys that doesn't match my model from the state dict
    pretrained_dict={k:v for k,v in model.features.state_dict().items() if f"features.{k}" in vgg16_pretrained_weights.keys()}

    conv1_weights=vgg16_pretrained_weights['features.0.weight']
    
    if model.input_channels != 3:
        conv1_weights=cross_modality_initialization(model.input_channels,conv1_weights)

    pretrained_dict['0.weight']=conv1_weights

    model.features.state_dict().update(pretrained_dict)
    model.features.load_state_dict(pretrained_dict)
