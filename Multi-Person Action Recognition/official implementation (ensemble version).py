import cv2
import torch
import numpy as np
from tracker import DeepSortTracker 
from model import *
from of_utils.prepare_flow import *
import copy
import os
import argparse
import torch.nn.functional as F


parser=argparse.ArgumentParser(description="Multi-Person Action Recognition tool")
parser.add_argument("--detector-confidence", dest="model_confidence",type=float, default=0.8, help="Detector confidence")
parser.add_argument("--detector-weights",dest="path_weights",type=str, help="Detector weights path",required=True)
parser.add_argument("--video-path",dest="path_video",type=str,help="Video path",required=True)
parser.add_argument("--action-weights",dest="spatiotemporal_weights",type=str,help="Path to the weights of the Action Recognition Network",required=True)
parser.add_argument("--retrieval",dest="folder_image_retrieval",default="Black list",type=str,help="Name of the retrieval folder (if not present, it will be created)")

args=parser.parse_args()


def calculate_IoU(bbox_1,bbox_2):
    x1_intersection=max(bbox_1[0],bbox_2[0])
    y1_intersection=max(bbox_1[1],bbox_2[1])
    x2_intersection=max(bbox_1[2],bbox_2[2])
    y2_intersection=max(bbox_1[3],bbox_2[3])

    area_intersection=max(0,x2_intersection-x1_intersection+1)*max(0,y2_intersection-y1_intersection+1)

    area1=(bbox_1[2]-bbox_1[0]+1)*(bbox_1[3]-bbox_1[1]+1)
    area2=(bbox_2[2]-bbox_2[0]+1)*(bbox_2[3]-bbox_2[1]+1)

    return area_intersection/(float(area1+area2-area_intersection))

def filter_by_IoU(list1,list2,threshold=0.5):
    results=[]

    if len(list1)==0 or len(list2)==0:
        return results

    for track_id_1, bbox_1 in list1:
        for track_id_2, bbox_2, presence in list2:
            if track_id_1==track_id_2 and presence is not False:
                IoU=calculate_IoU(bbox_1,bbox_2)
                if IoU > threshold:
                    results.append(track_id_2)

    return results

#These are used to know which action the person is performing
idx_to_class={0: 'kick', 1: 'punch', 2: 'run', 3: 'shoot_gun', 4: 'sword', 5: 'walk'}

#Parameters set by us (IT MUST BE SUBSTITUTED BY ARGS.PARSE)
MODEL_CONFIDENCE=args.model_confidence        #Accuracy of the bounding box (used as a threshold)
PATH_VIDEO=args.path_video                    #Test video path
PATH_WEIGHTS=args.path_weights                #Weights path

#Here I insert the name of the folder for the retrieval part
FOLDER_IMAGE_RETRIEVAL=args.folder_image_retrieval

#Spatiotemporal weights
SPATIOTEMPORAL_WEIGHTS=args.spatiotemporal_weights

#I extract the name of the video (it is used if you want to create a new video)
if '.' in PATH_VIDEO:
    VIDEO_NAME= '.'.join(PATH_VIDEO.split('.')[:-1])

#Loading Yolov7
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=torch.hub.load("WongKinYiu/yolov7","custom",f"{PATH_WEIGHTS}",trust_repo=True)

#Tracker initialization
tracker=DeepSortTracker()

#Action recognition model
spatiotemporal_model=TwoStreamArchitecture(num_classes=6).to(device)
spatiotemporal_model.load_state_dict(torch.load(SPATIOTEMPORAL_WEIGHTS,map_location=device),strict=True)

spatiotemporal_model.eval()

#I open the video
cap = cv2.VideoCapture(PATH_VIDEO)

#I extract the width, height and fps from the video
WIDTH=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS=int(cap.get(cv2.CAP_PROP_FPS))
TOTAL_FRAMES=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

RESIZE_RATE_WIDTH=320/WIDTH                       #Resize size of the windows
RESIZE_RATE_HEIGHT=240/HEIGHT

#If you want to create a new video
if FPS > 30:
    out=cv2.VideoWriter(f"{VIDEO_NAME}_tracking.mp4",cv2.VideoWriter_fourcc(*'mp4v'),int(FPS/2),(int(WIDTH),int(HEIGHT)))
else: 
    out=cv2.VideoWriter(f"{VIDEO_NAME}_tracking.mp4",cv2.VideoWriter_fourcc(*'mp4v'),int(FPS),(int(WIDTH),int(HEIGHT)))

#I create a folder called black_list (used for the retrieval part) if it is not present in the current directory
if not os.path.exists(FOLDER_IMAGE_RETRIEVAL):
    os.makedirs(FOLDER_IMAGE_RETRIEVAL)

T=2                        #It represents the temporal T taken
L=5                        #It represents the number of optical flow that i take
frame_to_take=4            #Used to know the first frame to be taken
flow_list=list()           #Used to keep the optical flow computed
frame_list=list()          #Used to keep the frame computed

bbox_dictionary=dict()     #Used to keep track of the bbox of the given track_id. track_id:[bbox_1,bbox_2] 

flow_index=0               #Used to insert flows into flow_tensor
output_twostream=11        #Used to know when we are ready to know the actions

frame_count=1              #Used to keep track of the current frame number

batches_track_id=list()    #Used to keep track of the track_id of the people that are performing an action.
last_track_id=list()       #Used to keep track of the last track_id. Used because the tracker assigns the indexes in a running way.

tmp=0

while cap.isOpened():
    #I extract the frames from the video
    success, frame = cap.read()

    if FPS > 30:
        tmp+=1

        if tmp%2 != 0:
            continue

    if success:
        if FPS > 30:
            print(f"Video under processing: {frame_count}/{int(TOTAL_FRAMES/2)} frames processed")
        else:
            print(f"Video under processing: {frame_count}/{int(TOTAL_FRAMES)} frames processed")
            
        annotated_frame=copy.deepcopy(frame)
        black_list_frame=copy.deepcopy(frame)
        frame=cv2.resize(frame,(int(WIDTH*RESIZE_RATE_WIDTH),int(HEIGHT*RESIZE_RATE_HEIGHT)))

        #I execute Yolov7 on the frame
        results = model(frame)

        #I only consider indexes = 0 (which represent a person) 
        classes=results.pandas().xywh[0]['class']==0
        results=results.pandas().xywh[0].to_numpy()[classes]
        boxes=results[:,0:4]
        conf=results[:,4]

        #Now i use MODEL_CONFIDENCE to filter the results (I consider only the ones that has confidence >= MODEL_CONFIDENCE)
        boxes=boxes[conf >= MODEL_CONFIDENCE]
        conf=conf[conf >= MODEL_CONFIDENCE]

        #I modify the data for the tracker. Data are accepted as: x,y(top-left),w,h
        boxes[:,0]=boxes[:,0]-boxes[:,2]/2
        boxes[:,1]=boxes[:,1]-boxes[:,3]/2

        if frame_count == 1:
            prev_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        else:
            gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            #Here I have computed the OF for the entire image, I now need to
            #use the bbox to extract 
            optical_flow=compute_TVL1(prev_frame,gray_frame)
            flow_list.append(copy.deepcopy(optical_flow))

        if frame_count == frame_to_take:
            tmp_img=copy.deepcopy(frame)
            frame_list.append(cv2.cvtColor(tmp_img,cv2.COLOR_BGR2RGB))

        tracker.update(frame,boxes,conf)
        for track in tracker.tracks:
            bbox=track.bbox
            x1,y1,x2,y2=bbox
            track_id=track.id
            
            #If the body goes outside the borders of the image, then i don't want to see the bounding box anymore
            #Errore se non ci sono persone allora non mostra nulla
            if not (x1 < 0 or x2 > WIDTH or y1 < 0 or y2 > HEIGHT):
                last_track_id.append((track_id,bbox))

                if frame_count == frame_to_take: 
                    if bbox_dictionary.get(track_id) is not None:
                        #Here I need to check if the bbox that we want to insert and
                        #the one already present refers to the same person (I do it by IoU > threshold)
                        if calculate_IoU(bbox_dictionary[track_id][0],bbox) > 0.7:
                            bbox_dictionary[track_id].append(bbox)
                    else:
                        bbox_dictionary[track_id]=list()
                        bbox_dictionary[track_id].append(bbox)

                if frame_count==output_twostream:
                    batches=len(list(filter(lambda x: len(x[1]) == T, bbox_dictionary.items())))
                    bbox_batches=[value for value in bbox_dictionary.values() if len(value)==2]
                    #The third element in the tuple is used to specify if the id has changed during 2 frames. If that happens
                    #the action is considered not performed anymore.
                    batches_track_id=[(key,value[1],True) for key,value in bbox_dictionary.items() if len(value)==2]

                    sono_label=list()
                    
                    for i in range(3):
                        lista=list()
                        for frame in frame_list:
                            if i==0:
                                lista.append(frame)
                            if i ==1:
                                lista.append(cv2.GaussianBlur(frame,(7,7),1))
                            if i==2:
                                lista.append(cv2.GaussianBlur(frame,(11,11),1.5))

                        flow_tensor=torch.FloatTensor(batches,10,T,130,70)      #Shape: (n, iC, T, H, W)
                        frame_tensor=torch.FloatTensor(batches,3,T,130,70)      #Shape: (n, iC, T, H, W)

                        
                        for num_batch in range(batches):
                            for t in range(T):
                                x_1,y_1,x_2,y_2=bbox_batches[num_batch][t]
                                frame_tensor[num_batch,:,t,:,:]=frame_transformation(lista[t][int(y_1):int(y_2),int(x_1):int(x_2)])

                                if t == 0:
                                    for num_flow in range(L):
                                        flow_tensor[num_batch,2*num_flow,t,:,:]=OF_transformation(flow_list[num_flow][:,:,0][int(y_1):int(y_2),int(x_1):int(x_2)])
                                        flow_tensor[num_batch,2*num_flow+1,t,:,:]=OF_transformation(flow_list[num_flow][:,:,1][int(y_1):int(y_2),int(x_1):int(x_2)])
                                elif t == 1:
                                    for num_flow in range(L,T*L):
                                        flow_tensor[num_batch,2*(num_flow-L),t,:,:]=OF_transformation(flow_list[num_flow][:,:,0][int(y_1):int(y_2),int(x_1):int(x_2)])
                                        flow_tensor[num_batch,2*(num_flow-L)+1,t,:,:]=OF_transformation(flow_list[num_flow][:,:,1][int(y_1):int(y_2),int(x_1):int(x_2)])
                                else:
                                    raise(ValueError, "The value of t should not be different from 1 and 2")

                        if batches != 0:     
                            with torch.no_grad():
                                labels=spatiotemporal_model(frame_tensor.to(device),flow_tensor.to(device))
                                labels=F.softmax(labels,dim=1)
                                sono_label.append(labels)
                    
                    #print(sono_label)
                    if len(sono_label) != 0:
                        mean_tensor=torch.zeros_like(sono_label[0])
                        for tensor in sono_label:
                            mean_tensor+=tensor
                        mean_tensor/=len(sono_label)
                        #print(mean_tensor)

                        _,pred=torch.max(mean_tensor,dim=1)
                        pred=pred.detach().cpu()
                        
                    bbox_dictionary=dict()
                    flow_list=list()
                    frame_list=list()
                    bbox_batches=list()
                    output_twostream+=5*T
            
                cv2.rectangle(annotated_frame,(int(x1*1/RESIZE_RATE_WIDTH),int(y1*1/RESIZE_RATE_HEIGHT)),(int(x2*1/RESIZE_RATE_WIDTH),int(y2*1/RESIZE_RATE_HEIGHT)),color=(255, 0, 0),thickness=int(3))
                frame=cv2.putText(annotated_frame, f"{track.id}", (int(x1*1/RESIZE_RATE_WIDTH)-10, int(y1*1/RESIZE_RATE_HEIGHT)+5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), int(3))

        #I perform all the task after because It might happen that the id
        #might be associated to another person after a few frames
        #(and thus I don't want to show anymore the action associated to the id)
        id_IoU=filter_by_IoU(last_track_id,batches_track_id)

        for i, (track_id,bbox,presence) in enumerate(batches_track_id):
            if track_id not in id_IoU:
                batches_track_id[i]=(track_id,bbox,False) 
        
        #Here depending on the type of action, i insert a bounding box with a different colour
        for id, bbox in last_track_id:
            for i,(id_batches,bbox_batches,presence) in enumerate(batches_track_id):
                if id_batches == id and presence == True:
                    if idx_to_class[pred[i].item()] == 'walk':
                        annotated_frame=cv2.rectangle(annotated_frame,(int(bbox[0]*1/RESIZE_RATE_WIDTH),int(bbox[1]*1/RESIZE_RATE_HEIGHT)),(int(bbox[2]*1/RESIZE_RATE_WIDTH),int(bbox[3]*1/RESIZE_RATE_HEIGHT)),color=(0, 128, 0),thickness=int(3))
                    elif idx_to_class[pred[i].item()] in ['kick','punch','shoot_gun','sword']:
                        if frame_count == (output_twostream-5*T):
                            cv2.imwrite(os.path.join(FOLDER_IMAGE_RETRIEVAL,f"{id_batches}_{frame_count}.jpg"),black_list_frame[int(bbox[1]*1/RESIZE_RATE_HEIGHT):int(bbox[3]*1/RESIZE_RATE_HEIGHT),int(bbox[0]*1/RESIZE_RATE_WIDTH):int(bbox[2]*1/RESIZE_RATE_WIDTH)],[cv2.IMWRITE_JPEG_QUALITY,100])
                        annotated_frame=cv2.rectangle(annotated_frame,(int(bbox[0]*1/RESIZE_RATE_WIDTH),int(bbox[1]*1/RESIZE_RATE_HEIGHT)),(int(bbox[2]*1/RESIZE_RATE_WIDTH),int(bbox[3]*1/RESIZE_RATE_HEIGHT)),color=(0, 0, 255),thickness=int(3))
                    else:
                        annotated_frame=cv2.rectangle(annotated_frame,(int(bbox[0]*1/RESIZE_RATE_WIDTH),int(bbox[1]*1/RESIZE_RATE_HEIGHT)),(int(bbox[2]*1/RESIZE_RATE_WIDTH),int(bbox[3]*1/RESIZE_RATE_HEIGHT)),color=(0, 165, 255),thickness=int(3))
                    annotated_frame=cv2.putText(annotated_frame, f"{id_batches}-{idx_to_class[pred[i].item()]}", (int(bbox[0]*1/RESIZE_RATE_WIDTH-10), int(bbox[1]*1/RESIZE_RATE_HEIGHT+5)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), int(3))
        
        if frame_count == frame_to_take:
            frame_to_take+=5

        last_track_id=list()

        pl=cv2.resize(annotated_frame,(WIDTH,HEIGHT))
        out.write(annotated_frame)
        
        if np.array_equal(annotated_frame,frame):
            cv2.imshow("Yolo",pl)
        else:
            cv2.imshow("Yolo", pl)
            
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        if frame_count > output_twostream:
            #Bisogna assicurarsi poi che quando entro nella zona tracker abbia effettivamente 10 elementi salvati
            output_twostream=frame_count+2*T
        if frame_count != 1:
            prev_frame=gray_frame
        frame_count+=1
    
    else:
        break


cap.release()
out.release()
cv2.destroyAllWindows()