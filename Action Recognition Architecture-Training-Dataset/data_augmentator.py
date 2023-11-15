import torchvision.transforms as T
import cv2
import os
import torch
import random
import numpy as np

"""
Here are described the quantity of videos in every action before performing Data Augmentation:
Videos < 40
    kick, sword
Video >= 40 and < 100
    punch, run, shoot-gun
Video >= 100
    walk
"""

def video_to_tensor(video_path):
    frames=[]
    video=cv2.VideoCapture(video_path)
    WIDTH=video.get(cv2.CAP_PROP_FRAME_WIDTH)
    HEIGHT=video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    while True:
        success, frame=video.read()
        if not success:
            break

        frame_tensor=T.ToTensor()(frame).unsqueeze(0)
        
        frames.append(frame_tensor)
    video.release()

    return WIDTH,HEIGHT,torch.cat(frames)

def augment_video(file_path,name_file, output_video_folder):
    """
        Here augmentation is performed on the entire video, the following represents
        the augmentation parameters used:
            -ColorJitter: brightness, contrast, saturation and hue are modified.
            -RandomHorizontalFlip: video frames have a probability of 50% to be flipped orizontally
            -RandomErasing: 2 random erasing are introduced onto the image. They have a probability
                            to pop-up of 70% the first one, and 50% the second one.
            -RandomRotation: video frames have a random rotation from -20° to 20°.
    """
    VIDEO_WIDTH,VIDEO_HEIGHT,video_tensor=video_to_tensor(file_path)
    
    trasform=T.Compose([T.ColorJitter(brightness=(0.5,1.5),contrast=1.5,saturation=2,hue=0.08),
                        T.RandomHorizontalFlip(p=0.5),
                        T.RandomErasing(p=0.7,scale=(0.02,0.02),ratio=(0.3,3.3)),
                        T.RandomErasing(p=0.5,scale=(0.02,0.02),ratio=(0.3,3.3)),
                        T.RandomRotation((-10,10),interpolation=T.InterpolationMode.BILINEAR)])
    
    trasformed_video_tensor=trasform(video_tensor)
    #print(trasformed_video_tensor.shape)
    
    video_out=cv2.VideoWriter(os.path.join(output_video_folder,name_file+".avi"),cv2.VideoWriter_fourcc(*"XVID"),
                              30,(int(VIDEO_WIDTH),int(VIDEO_HEIGHT)))
    for idx,frame_tensor in enumerate(trasformed_video_tensor):
        frame=frame_tensor.permute(1,2,0).numpy()
        frame=(frame*255).astype(np.uint8)
        video_out.write(frame)
    video_out.release()

def folder(input_folder, output_folder):
    #If the folder doesn't exist, i create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for folder_name in os.listdir(input_folder):
        folder_path=os.path.join(input_folder,folder_name)

        if not os.path.isdir(folder_path):
            continue

        #I create a folder with the name of folder_name
        output_subfolder=os.path.join(output_folder,folder_name)
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

        for file_name in os.listdir(folder_path):
            file_path=os.path.join(folder_path,file_name)

            name_file="2_Augmented_"+os.path.splitext(file_name)[0]
            augment_video(file_path,name_file,output_subfolder)

#Here the input and output folder are defined
input_folder='HMDB51'
output_folder='HMDB51_augmented_2'

folder(input_folder,output_folder)

