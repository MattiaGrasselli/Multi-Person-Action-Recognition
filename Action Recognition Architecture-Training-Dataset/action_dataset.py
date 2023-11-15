import os
import cv2
import torch
import numpy as np
import time

MODEL_CONFIDENCE=0.8        #Accuracy of the bounding boxes (used as threshold)
PATH_WEIGHTS="best.pt"      #Path for the Yolov7 weights
RESIZE_RATE=0.2             #Resize rate for the visualization window

EXTRACT_FRAME_NUMBER=5      #Questo valore indica ogni quanto si estrae 1 frame

H=0
W=0
bbox_number=0

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

def mean_H_W(boxes):
    """
        The following function is used for storing H and W of each bounding box.
        This information is used to know which is the average H and W in the entire
        dataset.
    """
    global H,W,bbox_number

    for box in boxes:
        H=H+box[3]
        W=W+box[2]
        bbox_number=bbox_number+1

def extract_bbox(frame):
    """
        The following function is used to extract the bouding boxes.
    """
    #I pass the frame to the Yolov7 model
    results = model(frame)
    #I will consider only the indexes with class=0 (which represents the people)
    classes=results.pandas().xywh[0]['class']==0
    results=results.pandas().xywh[0].to_numpy()[classes]
    
    boxes=results[:,0:4]
    #I modify the data so that they get the following form: x,y(top-left),w,h
    boxes[:,0]=boxes[:,0]-boxes[:,2]/2
    boxes[:,1]=boxes[:,1]-boxes[:,3]/2

    return boxes

def save_images(save_folder,frame,boxes,frame_count,width,height):
    """
        The following function is used for saving the images that represent
        the person:
            -save_folder: folder where the image will be saved.
            -frame: it can either be a frame or a optical flow frame.
            -boxes: bounding box coordinates.
            -frame_count: index used for storing different bounding boxes.
            -width: width of the frame.
            -height: height of the frame.
    """
    numero=0
    
    for box in boxes:
        x1,y1,w,h=box
        #print(f"Questo è la WIDTH e HEIGHT: {WIDTH}, {HEIGHT}")"
        if not (x1 < 0 or x1+w > width or y1 < 0 or y1+h > height):
            #I extract the image using the bouding box
            cropped_img=frame[int(y1):int(y1+h),int(x1):int(x1+w)]

            if numero == 0:
                frame_path=os.path.join(save_folder[numero],f"{frame_count}.jpg")
            else:
                frame_path=os.path.join(save_folder[numero],f"{frame_count}_{numero}.jpg")
            numero=numero+1

            cv2.imwrite(frame_path,cropped_img,[cv2.IMWRITE_JPEG_QUALITY,100])


def extract_frames(video_path, output_folder,output_folder_flow):
    video = cv2.VideoCapture(video_path)
    
    WIDTH=int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT=int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS=int(video.get(cv2.CAP_PROP_FPS))

    frame_flow=list()
    frame_to_be_saved=None

    i_taken=True
    directory_number=1

    if video.isOpened():
            frame_count=1

            frame_to_take=4
            directory_change=6
            
            while True:
                success,frame=video.read()
                if not success:
                    break
                
                if frame_count == 1:
                    #Convert the image in grayscale
                    prev_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    frame_count += 1
                    continue

                frame_OF=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

                frame_flow.append([prev_frame,frame_OF])

                if not i_taken:
                    boxes=extract_bbox(frame)
                    mean_H_W(boxes)
                    i_taken=True
                
                if frame_count==frame_to_take:
                    boxes=extract_bbox(frame)
                    if boxes.shape[0] == 0:
                        i_taken=False
                    mean_H_W(boxes)

                    frame_to_be_saved=frame
                    frame_to_take+=5

                #print(f"Questo e\' frame_count: {frame_count}.Mentre questo è directory_change: {directory_change}")
                if directory_change == frame_count:
                    tmp_output_folder=list()
                    tmp_output_folder_flow=list()
                    for box in boxes:
                        x1,y1,w,h=box
                        #print(f"Questo è la WIDTH e HEIGHT: {WIDTH}, {HEIGHT}")"
                        if not (x1 < 0 or x1+w > WIDTH or y1 < 0 or y1+h > HEIGHT):
                            tmp_output_folder.append(f"{output_folder}_{directory_number}")
                            tmp_output_folder_flow.append(f"{output_folder_flow}_{directory_number}")
                            if not os.path.exists(f"{output_folder}_{directory_number}"):
                                os.makedirs(f"{output_folder}_{directory_number}")
                            if not os.path.exists(f"{output_folder_flow}_{directory_number}"):
                                os.makedirs(f"{output_folder_flow}_{directory_number}")
                            
                            directory_number+=1

                    if boxes.shape[0] != 0:
                        for i in range(5):
                            flow=compute_TVL1(frame_flow[i][0],frame_flow[i][1])
                            save_images(tmp_output_folder_flow,flow,boxes,i+1,WIDTH,HEIGHT)
                        save_images(tmp_output_folder,frame_to_be_saved, boxes,frame_count,WIDTH,HEIGHT)

                    frame_flow=list()
                    directory_change += 5

                prev_frame=frame_OF
                frame_count += 1

    video.release()

def folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    global H,W,bbox_number

    for folder_name in os.listdir(input_folder):
        folder_path=os.path.join(input_folder,folder_name)

        if not os.path.isdir(folder_path):
            continue

        output_subfolder=os.path.join(output_folder,folder_name)
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)


        for file_name in os.listdir(folder_path):
            file_path=os.path.join(folder_path,file_name)

            output_video_folder=os.path.join(output_subfolder,os.path.splitext(file_name)[0])
            print(f"Questo è l'output_video_folder: {output_video_folder}")
            output_video_folder_rgbflow="flow_"+output_video_folder
            print(f"Questo è l'output video folder flow: {output_video_folder_rgbflow}")
            extract_frames(file_path,output_video_folder,output_video_folder_rgbflow)

    with open('mean_H_W', "w") as file:
        #I write the mean of H and W casted to the value below
        H_mean=H/bbox_number
        W_mean=W/bbox_number

        #I write H and W in the file
        file.write(str(H_mean))
        file.write('\n')
        file.write(str(W_mean))

    print("File mean_H_W costruto")

input_folder='HMDB51_augmented_2'
output_folder_frames='HMDB51_Dataset_2'

#I load Yolov7
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=torch.hub.load("WongKinYiu/yolov7","custom",f"{PATH_WEIGHTS}",trust_repo=True)

folder(input_folder,output_folder_frames)