import os
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import numpy as np

idx_to_class={0: 'kick', 1: 'punch', 2: 'run', 3: 'shoot_gun', 4: 'sword', 5: 'walk'}

def PrepareOF(opt_root,opt_files_sorted,num_opt_flow,transform):
    """
        This function has the purpose of extracting the stored num_opt_flow 
        optical flow and it creates a tensor that has the gradient x and y
        from each optical flow concatenated over the tensor channels. 
    """
    for i in range(num_opt_flow):
        #print(f"Ti printo 2*(i-1): {2*i}")
        #print(f"Ti printo 2*(i-1)+1: {2*i+1}")
        first_opt=os.path.join(opt_root,opt_files_sorted[i])
        opt=cv2.imread(first_opt)
        
        if i == 0:
            #optical_flow is a tensor (iC,H,W)
            optical_flow=torch.FloatTensor(num_opt_flow*2,130,70)

        #opt[:,:,0] represents the x gradient
        optical_flow[2*i,:,:]=transform(opt[:,:,0])
        #opt[:,:,1] represents the y gradient
        optical_flow[2*i+1,:,:]=transform(opt[:,:,1])

    return optical_flow

class HMDB51dataset(Dataset):
    def __init__(self,csv_file,root_dir,transform=transforms.ToTensor()) -> None:
        self.info=pd.read_csv(csv_file,names=['file_name','label'],delimiter=';')

        self.root_dir=root_dir
        self.transform=transform

    def __len__(self):
        return len(self.info)
     
class SpatialDataset(HMDB51dataset):
    def __init__(self,csv_file,root_dir,num_frame=1,transform=transforms.ToTensor()) -> None:
        super().__init__(csv_file,root_dir,transform)
        self.num_frame=num_frame
    
    def __getitem__(self,idx):
        """
            This function is used for giving as a result the sample with the name
            name_idx and its label.
        """
        assert isinstance(idx,int)

        label=self.info.iloc[idx]['label']
        name_idx=self.info.iloc[idx]['file_name']

        class_name=idx_to_class[label]

        root_name=f"HMDB51_Dataset{os.sep}{class_name}{os.sep}{name_idx}"
        img_name=os.path.join(self.root_dir, root_name)
        file=os.listdir(img_name)

        assert len(file) == self.num_frame

        img_name=os.path.join(img_name,file[0])
        frame=cv2.imread(img_name)
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        frame=self.transform(frame)

        if '_med_' in name_idx:
            gaussian_blur=transforms.GaussianBlur((7,7),sigma=1.0)
        if '_bad_' in name_idx:
            gaussian_blur=transforms.GaussianBlur((11,11),sigma=1.5)

        if '_med_' in name_idx or '_bad_' in name_idx:
            frame=gaussian_blur(frame)
        
        return frame, label

class TemporalDataset(HMDB51dataset):
    def __init__(self,csv_file,root_dir,num_opt_flow=5,transform=transforms.ToTensor()) -> None:
        super().__init__(csv_file,root_dir,transform)
        self.num_opt_flow=num_opt_flow

    def __getitem__(self,idx):
        """
            This function is used for giving as a result the sample with the name
            name_idx and its label.
        """
        assert isinstance(idx,int)

        label=self.info.iloc[idx]['label']
        name_idx=self.info.iloc[idx]['file_name']

        class_name=idx_to_class[label]

        root_name=f"flow_HMDB51_Dataset{os.sep}{class_name}{os.sep}{name_idx}"
        img_name=os.path.join(self.root_dir,root_name)
        file=os.listdir(img_name)
        file.sort()

        assert len(file) == self.num_opt_flow
        
        optical_flow=PrepareOF(img_name,file,self.num_opt_flow,self.transform)

        return optical_flow, label

class SpatioTemporalDataset(HMDB51dataset):
    def __init__(self,csv_file,root_dir,num_frame=1,num_opt_flow=5,spatial_transform=transforms.ToTensor(),
                 temporal_transform=transforms.ToTensor()) -> None:
        super().__init__(csv_file,root_dir)
        #Since in this case i have 2 transform, self.transform inherited from HMDB51Dataset
        #will be considered as spatial_transform and self.temporal_transform will contain temporal_transform 
        self.transform=spatial_transform
        self.temporal_transform=temporal_transform

        self.num_frame=num_frame
        self.num_opt_flow=num_opt_flow

    def __getitem__(self,idx):
        """
            This function is used for giving as a result the sample with the name
            name_idx and its label.
        """
        assert isinstance(idx,int)

        label=self.info.iloc[idx]['label']
        name_idx=self.info.iloc[idx]['file_name']

        class_name=idx_to_class[label]

        root_name=f"Spatiotemporal_HMDB51{os.sep}{class_name}{os.sep}{name_idx}"
        frame_root=os.path.join(self.root_dir,root_name)
        opt_root=os.path.join(self.root_dir,"flow_"+root_name)

        file=os.listdir(frame_root)

        frame=torch.FloatTensor(3,2,130,70)

        for i in range(len(file)):
            img_name=os.path.join(frame_root,file[i])
            tmp=cv2.imread(img_name)
            tmp=cv2.cvtColor(tmp,cv2.COLOR_BGR2RGB)
            
            tmp=self.transform(tmp)

            if '_med_' in img_name:
                gaussian_blur=transforms.GaussianBlur((7,7),sigma=1.0)
            if '_bad_' in img_name:
                gaussian_blur=transforms.GaussianBlur((11,11),sigma=1.5)
    
            if '_med_' in img_name or '_bad_' in img_name:
                tmp=gaussian_blur(tmp)
            
            frame[:,i,:,:]=tmp

        assert len(file) == 2*self.num_frame

        opt_file=os.listdir(opt_root)
        opt_file.sort(key=mySort)

        assert len(opt_file) == 2*self.num_opt_flow
        
        optical_flow=torch.FloatTensor(10,2,130,70)

        #print(opt_file[0:5])
        #print(opt_file[5:10])
        optical_flow[:,0,:,:]=PrepareOF(opt_root,opt_file[:5],self.num_opt_flow,self.temporal_transform)
        optical_flow[:,1,:,:]=PrepareOF(opt_root,opt_file[5:10],self.num_opt_flow,self.temporal_transform)

        return frame, optical_flow, label

# IMPORTANT NOTE: the assumption is that the file are numbered correctly (for instance. '1.jpg', '2.jpg' and so on) 
def mySort(element):
    return int(element.split('.')[0])

"""
temporal_transforms=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((130,70))
])
spatial_transforms=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((130,70)),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])    
])

#pippo=SpatialDataset('C:\\Users\\grass\\Desktop\\Action Recognition Network\\esempio_funzionamento\\file_csv_per_testing\\file_csv.csv','C:\\Users\\grass\\Desktop\\Action Recognition Network\\esempio_funzionamento\\dataset_esempio2',transform=spatial_transforms)
#print(pippo[1])

pippo=SpatioTemporalDataset('C:\\Users\\grass\\Desktop\\Action Recognition Network\\esempio_funzionamento\\file_csv_per_testing\\file_csv.csv','C:\\Users\\grass\\Desktop\\Action Recognition Network\\esempio_funzionamento\\dataset_esempio',temporal_transform=temporal_transforms,spatial_transform=spatial_transforms)
print(pippo[0])
"""