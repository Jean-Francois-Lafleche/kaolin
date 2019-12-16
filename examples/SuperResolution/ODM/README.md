# Example: ODM-superresolution 

These example scripts lets you train a voxel Super-Resolution network on the ShapeNet Dataset using Orthographic Depth Maps(ODMS). For details on the algorithms see: "Multi-View Silhouette and Depth Decomposition for High Resolution 3D Object Representation" : https://arxiv.org/abs/1802.09987. 


DELETE There are two training schemes, one which Directly predicts superresolves ODMs (Direct), and one which separates the problem into occupancy and residual predictions (MVD). 


### Training the network: Direct

To train using Direct run
```
python train_Direct.py
```


### Evaluating the network: Direct

To evaluate a trained Direct model run 
```
python eval_Direct.py
```


### Training the network: MVD

To train using MVD run
```
python train_MVD.py
```


### Evaluating the network: MVD

To evaluate a trained MVD model run 
```
python eval_MVD.py
```