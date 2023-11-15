import numpy as np
import cv2
import torch
from torchvision.transforms import transforms

# Here i create the transform
spatial_transforms=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((130,70)),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])    
])
temporal_transforms=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((130,70)),
    transforms.Normalize(mean=[0.5007],std=[0.0221])
])

def compute_TVL1(prev, curr, bound=20):
    """
        Compute the TV-L1 optical flow.
        The operations here follows the settings of the Temporal Segment Network and the code
        in denseflow.
        bound: limit the maximum movement of one pixel. It's an optional setting.
    """
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)
    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2*bound))
    flow[flow <= 0] = 0
    flow[flow >= 255] = 255

    flow = np.concatenate((flow, np.zeros((flow.shape[0],flow.shape[1],1))), axis=2)

    flow = np.round(flow).astype(int)
    return flow

def OF_transformation(optical_flow):
    return temporal_transforms(np.uint8(optical_flow))

def frame_transformation(frame):
    return spatial_transforms(frame)