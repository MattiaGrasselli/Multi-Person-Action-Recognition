# Retrieval Component

## Installation
```
git clone https://github.com/timesler/facenet-pytorch.git facenet_pytorch
conda create -n retrieval_env python=3.9
conda activate retrieval_env
pip install -r requirements.txt
```

## Usage
Evaluate different configurations of the HOG detector on the WIDER FACE dataset
```
python wider_face_predictions.py --wider-root ./WIDER_val/images --dest-dir ./wider_predictions_u2 --upsample 2
```
Then generate plots with `wider_eval.m`

To run the retrieval component:
```
python -m retrieval.py --source "data/sequence02.mov" --blacklist-dir "blacklists/blacklist01" --upsample 1
```