# CHARM: Cryosection Histopathology Assessment and Review Machine (TBD) 
![](figures/charm-workflow.png)


## Pre-requisites:
* Linux (Tested on Ubuntu 18.04)
* NVIDIA GPU (Tested on Nvidia GeForce RTX 6000/8000/2080Ti)
* Python (3.7+).
* Check requirement.txt under pytorch and tf folders respectively for other dependences. 

### Data Preparation
For each cryosection whole slide image, we tiled it in 1000 by 1000 pixels with 50% overlapping in both x and y directions using [level 0] 40x magnification. 
#### Partitioned dataset structure (csv and h5 file)
We saved all tiles for each task in a h5 file. In the h5 files, tile patches were saved as numpy array. Both labls for subtask and patient ID were provided correspondingly. We randomly partitioned the datasets based on patient IDs into training, validation and testing sets in 70/15/15 percents. Then, we used a csv file to record the each partition with three columns: "Patient ID", "Label" and "par" (train/val/test). 

### Training
* Vision transformers were trained using pytorch framework and timm (https://github.com/rwightman/pytorch-image-models).
  Install all dependencies:
  ``` shell
  pip install -r requirements.txt
  ```
* CNNs were trained using tensorflow framework.




### Validation

### Issues
### License 
© [HMS DBMI Yu Lab](https://yulab.hms.harvard.edu/) - This code is made available under the GPLv3 License and is available for non-commercial academic purposes. 

### Acknowledge and reference


