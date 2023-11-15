from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections
from deep_sort.application_util import preprocessing

class DeepTrack:
    def __init__(self,id,bbox):
        self.id=id
        self.bbox=bbox

class DeepSortTracker:
    def __init__(self,max_cosine_distance=0.2,nn_budget=None,model_filename="Re-id/mars-small128.pb"):
        self._max_cosine_distance=max_cosine_distance
        self._nn_budget=nn_budget
        self._model_filename=model_filename

        self.metric=nn_matching.NearestNeighborDistanceMetric("cosine",self._max_cosine_distance,self._nn_budget)
        self.tracker=Tracker(self.metric)
        self.encoder=generate_detections.create_box_encoder(self._model_filename,batch_size=1)

    def update(self,img,bboxes,confs):
        #I extract the features
        features=self.encoder(img,bboxes)

        detections=[Detection(bbox,conf,feature) for bbox,conf,feature in zip(bboxes,confs,features)]

        self.tracker.predict()
        self.tracker.update(detections)

        self.tracks=list()
        
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            self.tracks.append(DeepTrack(track.track_id,track.to_tlbr()))
