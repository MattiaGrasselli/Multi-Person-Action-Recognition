from VGG16 import *
import torch.nn as nn 
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TwoStreamArchitecture(nn.Module):
    def __init__(self,spatial_stream_weights=None,temporal_stream_weights=None,num_classes=1000) -> None:
        super().__init__()
        
        self.SpatialStream=VGG16(num_classes=num_classes)
        if spatial_stream_weights is not None:
            self.SpatialStream.load_state_dict(torch.load(spatial_stream_weights),strict=True)

        self.TemporalStream=VGG16(input_channels=10,num_classes=num_classes)
        if temporal_stream_weights is not None:
            self.TemporalStream.load_state_dict(torch.load(temporal_stream_weights),strict=True)

        #I remove the last 4 layers because it is where the connection between the 2 networks will happen
        self.SpatialStream=nn.Sequential(*(list(self.SpatialStream.features.children())[:-1]))
        self.TemporalStream=nn.Sequential(*(list(self.TemporalStream.features.children())[:-1]))

        #By setting param.require_grad to False, i'm freezing those layers
        for param in self.SpatialStream.parameters():
            param.requires_grad=False
        
        for param in self.TemporalStream.parameters():
            param.requires_grad=False
    
        self.conv3d=nn.Conv3d(in_channels=1024,out_channels=512,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1))
        self.relu3d=nn.ReLU()
        self.maxPool3d=nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2))

        self.AdaptiveAvgPool=nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc=nn.Linear(in_features=512,out_features=num_classes,bias=True)

    def forward(self,x,y):        
        # out1 and out2 tensor has the following shape (n,iC,T,H,W)
        out1=torch.FloatTensor(x.shape[0],512,x.shape[2],8,4).to(device)
        out2=torch.FloatTensor(y.shape[0],512,y.shape[2],8,4).to(device)

        for n in range(x.shape[0]):
            for t in range(x.shape[2]):
                out1[n,:,t,:,:]=self.SpatialStream(torch.unsqueeze(x[n,:,t,:,:],0))
                out2[n,:,t,:,:]=self.TemporalStream(torch.unsqueeze(y[n,:,t,:,:],0))
        
        #print(out1.shape)
        #print(out2.shape)
        out=self._fusion(out1,out2).to(device)

        #print(out.shape)
        out=self.conv3d(out)
        out=self.relu3d(out)
        out=self.maxPool3d(out)

        #print(out.shape)
        out=self.AdaptiveAvgPool(out)

        out = out.view(out.size(0), -1)

        #print(out.shape)
        out=F.dropout(out,p=0.5,training=self.training)
        #print(out.shape)
        out = self.fc(out)

        return out
    
    def _fusion(self,spatial,temporal):
        assert spatial.shape == temporal.shape
        # Concat will have a shape (n,2*iC,T,H,W)
        concat=torch.FloatTensor(spatial.shape[0],2*spatial.shape[1],spatial.shape[2],spatial.shape[3],spatial.shape[4])

        # This could proably be refactored in a better way
        for n in range(spatial.shape[0]):
            for t in range(spatial.shape[2]):
                for k in range(spatial.shape[1]):
                    concat[n,2*k,t,:,:]=spatial[n,k,t,:,:]
                    concat[n,2*k+1,t,:,:]=temporal[n,k,t,:,:]
    
        return concat
    