from VGG16 import *
from dataset import *
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model import *
import numpy as np

parser=argparse.ArgumentParser(description="Spatio-temporal stream train")
parser.add_argument("--train-csv", dest="training_csv", help="Path to the train csv")
parser.add_argument("--validation-csv", dest="validation_csv",help="Path to the validation csv")
parser.add_argument("--training-root", dest="training_root",help="Path to the training dataset")
parser.add_argument("--validation-root", dest="validation_root",help="Path to the validation dataset")
parser.add_argument("--spatial-weights",dest="spatial_stream_weights",help="Path to the spatial stream weights")
parser.add_argument("--temporal-weights",dest="temporal_stream_weights",help="Path to the temporal stream weights")
parser.add_argument("--epochs", dest="N_EPOCH",type=int, help="Total number of epochs")
parser.add_argument("--batch-size", dest="BATCH_SIZE",type=int, help="Training Batch-size. \
                    Validation and Test batches will be a half of the training one")
parser.add_argument("--learning-rate", dest="LEARNING_RATE",type=float, help="Learning rate")
parser.add_argument("--momentum",dest="MOMENTUM",type=float, help="Momentum")
parser.add_argument("--weight-decay", dest="WEIGHT_DECAY",type=float, help="Weight-Decay")
parser.add_argument("--num-workers",dest="NUM_WORKERS",type=int, help="Number of workers")
parser.add_argument("--early-stopping",dest="EARLY_STOPPING",type=int,help="After early_stopping \
                    epochs without better loss, Early Stopping happens")
parser.add_argument("--save-root", dest="save_root",type=str, help="Save root where weights will be saved")

args=parser.parse_args()


SAVE_ROOT=args.save_root
N_EPOCH=args.N_EPOCH
BATCH_SIZE=args.BATCH_SIZE
LEARNING_RATE=args.LEARNING_RATE
MOMENTUM=args.MOMENTUM
WEIGHT_DECAY=args.WEIGHT_DECAY
NUM_WORKERS=args.NUM_WORKERS
EARLY_STOPPING=args.EARLY_STOPPING

#Here i specify the file_csv names for training, validation and test
training_csv=args.training_csv
validation_csv=args.validation_csv

#Dataset roots
training_root=args.training_root
validation_root=args.validation_root

#Weights
spatial_stream_weights=args.spatial_stream_weights
temporal_stream_weights=args.temporal_stream_weights

#Model definition
model=TwoStreamArchitecture(spatial_stream_weights,temporal_stream_weights,num_classes=6)

#print(summary(model,[(3,2,130,70),(10,2,130,70)],batch_size=3,device='cuda'))

#Check which are the parameters trainable
"""
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name},{param.name}")
"""

#Here i create the transform
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

#Here Datasets are defined
training_dataset=SpatioTemporalDataset(training_csv,training_root,spatial_transform=spatial_transforms,
                                       temporal_transform=temporal_transforms)
validation_dataset=SpatioTemporalDataset(validation_csv,validation_root,spatial_transform=spatial_transforms,
                                         temporal_transform=temporal_transforms)


#Here there are the DataLoaders
training_loader=DataLoader(training_dataset,BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS)
validation_loader=DataLoader(validation_dataset,BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS)

if torch.cuda.is_available():
    model.cuda()

#Here the loss function is definined
loss_function=nn.CrossEntropyLoss()

model_parameters=filter(lambda p: p.requires_grad, model.parameters())
opt=torch.optim.SGD(model_parameters,lr=LEARNING_RATE,momentum=MOMENTUM,weight_decay=WEIGHT_DECAY)

if __name__=='__main__':
    min_loss=np.Inf
    is_early_stopping_happened=False


    metrics=[]

    #Training and validation is performed
    for epoch in range(0, N_EPOCH):
        print(f"This is the epoch number: {epoch+1}")

        mean_train_loss=0.0
        mean_train_accuracy=0.0
        
        mean_validation_loss=0.0
        mean_validation_accuracy=0.0

        #-->Train is performed<--
        model.train()

        #IMPORTANT NOTE: frame, optical_flow will have the following shape: (n, iC, T, H ,W)
        for batch_id, (frame, optical_flow, label) in enumerate(training_loader):
            #print(f"Sono l\'input: {input}")
            #print(f"Sono la label: {label}")
            if torch.cuda.is_available():
                frame,optical_flow,label=frame.cuda(),optical_flow.cuda(),label.cuda()

            #Set the gradient to 0
            opt.zero_grad()

            #Forward-pass
            label_pred=model(frame,optical_flow)

            #Calculate the loss
            loss=loss_function(label_pred,label)

            #Backward-pass
            loss.backward()

            #Update parameters
            opt.step()

            #Track train loss by multiplying the average loss by the number of examples in the batch
            mean_train_loss += loss.item()*frame.size(0) 

            _,pred=torch.max(label_pred,dim=1)
            correct_tensor=pred.eq(label.data.view_as(pred))
            accuracy=100*torch.mean(correct_tensor.type(torch.FloatTensor))
            mean_train_accuracy += accuracy.item() * frame.size(0)
            #print(f"This is the prediction done: {pred}")
            #print(f"This is the correct tensor: {correct_tensor}")
            #print(f"This is the accuracy: {accuracy}")

        #Compute average accuracy during training
        mean_train_accuracy=mean_train_accuracy/len(training_loader.dataset)
        #Compute average loss
        mean_train_loss=mean_train_loss/len(training_loader.dataset)


        #-->Validation is performed<--
        model.eval()

        for frame, optical_flow, labels in validation_loader:
            #print(f"This is the value of inputs: {inputs.shape}")
            #print(f"This is the value of labels: {labels}")
            
            if torch.cuda.is_available():
                frame,optical_flow,labels=frame.cuda(),optical_flow.cuda(),labels.cuda()

            #Forward Pass
            label_pred=model(frame,optical_flow)

            #Validation Loss
            loss=loss_function(label_pred,labels)
            #I multiply the loss with the number of examples in the batch
            mean_validation_loss+=loss.item()*frame.size(0)

            #Compute validation accuracy
            _,pred=torch.max(label_pred.data,1)
            correct_tensor=pred.eq(labels.data.view_as(pred))
            accuracy=100*torch.mean(correct_tensor.type(torch.FloatTensor))

            mean_validation_accuracy += accuracy.item()*frame.size(0) 

            #print(f"This is the prediction done: {pred}")
            #print(f"This is the correct tensor: {correct_tensor}")
            #print(f"This is the accuracy: {accuracy}")

        #Average Loss
        mean_validation_loss=mean_validation_loss/len(validation_loader.dataset)
        #Average Accuracy
        mean_validation_accuracy=mean_validation_accuracy/len(validation_loader.dataset)

        #print(f"This is the mean loss: {mean_validation_loss}")
        #print(f"This is the mean accuracy: {mean_validation_accuracy}")
        
        metrics.append([mean_train_accuracy,mean_train_loss,mean_validation_accuracy,mean_validation_loss])
        
        #I save the metrics in a csv file
        metrics_to_save=pd.DataFrame(metrics,columns=['mean_train_accuracy',
                            'mean_train_loss','mean_validation_accuracy','mean_validation_loss'])

        metrics_to_save.to_csv(f'{SAVE_ROOT}/spatiotemporal_train_validation_metrics.csv',sep=";")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': mean_validation_loss
        },os.path.join(SAVE_ROOT,"last_spatiotemporal_checkpoint.pth.tar"))

        if mean_validation_loss <  min_loss:
            torch.save(model.state_dict(),os.path.join(SAVE_ROOT,"spatiotemporal_best.pt"))

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': mean_validation_loss
            },os.path.join(SAVE_ROOT,"spatiotemporal_checkpoint.pth.tar"))
            
            print(metrics)

            min_loss=mean_validation_loss
            best_epoch=epoch
            no_improvement=0
        #The following will represent an Early Stopping
        else:
            no_improvement+=1
            if no_improvement >= EARLY_STOPPING:
                print(f"Early stopping has happened. The best epoch: {best_epoch} with loss: {min_loss}.\nTotal number of epoch performed before Early Stopping: {epoch}")
                is_early_stopping_happened=True
                break
