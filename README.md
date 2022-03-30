# CHARM: Cryosection Histopathology Assessment and Review Machine
Machine Learning for Cryosection Pathology Predicts the 2021 WHO Classification of Glioma


![](figures/charm-workflow.png)


## Pre-requisites:
* Linux (Tested on Ubuntu 18.04)
* NVIDIA GPU (Tested on Nvidia GeForce RTX 6000/8000/2080Ti)
* Python (3.7+), TensorFlow (2.3.0).
* Cuda 10.2
* Check requirement.txt under pytorch and tf folders respectively for other dependencies. 

### Data Preparation
For each cryosection whole slide image, we tiled it in 1000 by 1000 pixels with 50% overlapping in both x and y directions using [level 0] 40x magnification. 
#### Partitioned dataset structure (csv and h5 file)
We saved all pathology images for each task in an h5 file. In the h5 files, tiled patches were saved as numpy array. The labels for each subtask and patient ID were provided in the array. We randomly partitioned the datasets based on patient IDs into training, validation, and testing sets in 70/15/15 percent. Then, we used a csv file to record each partition with three columns: "Patient ID", "Label" and "par" (train/val/test). 

### Training and validation
* Vision transformers were trained using the PyTorch framework and timm (https://github.com/rwightman/pytorch-image-models).
  Install all dependencies:
  ``` shell
  pip install -r requirements.txt
  ```
  #### usage example: IDH-LGG classification. 
  ##### Training
  *-- device: how many gpus are in use. 
  *-- loss: e.g. class-balanced loss or cross-entropy loss
  *-- model: e.g. 'SWIN' ---> swin transformer  
  *-- namemarker:  {model name}-Meta-csv-{task name}-{dataset name}-{partition csv ID}. e.g. 'SWIN-Meta-csv-IDH-LGG-0'
  ``` shell
  python main.py --device=8 --loss "CE" --model "SWIN" --outdim=2 --batch=100 --lr=0.001 --metric 'AUC' --namemarker "SWIN-622-Meta-csv-IDH-LGG-0" --optimizer 'sgd' --end-epoch=10 --savekey /FINAL/IDH_LGG/ --csv /partitions/par70_15_15/ --path /processed_data/task3/IDHdetection_LGG_BP.hdf5> logs/FINAL-CHECK/LGG-0.log 2>&1
  ```
   ##### Testing
     ``` shell
  python main.py --onlytest --model "SWIN" --batch=150 --outdim=2 --metric 'AUC' --namemarker "SWIN-224-Meta-csv-IDH-LGG-0" --modelpath /FINAL/IDH_LGG/0.0001_100_E200_ViT-Swin_CB_0.9999_class_AUC_1.0_noaug__nonconst_final-SWIN-224-Meta-csv-IDH-LGG-0.pt --path /processed_data/task3/IDHdetection_LGG_BP.hdf5 --csv /processed_data/par70_15_15/> logs/FINAL-CHECK/LGG-SWIN-IDH-BP-0-TEST.log 2>&1
  ```
  
* CNNs were built using TensorFlow 2.3.0.
  #### Example data
    Example csv files for training and testing the models, especially for running TMB regression, were placed in the tensorflow2 folder.
  
  #### Data names and locations
    All data was assumed to be placed under './data'. The hdf5 files storing images were assumed to be name as "{task_name}_{ds_name}.hdf5" where task_name could be 
    "cancerVsBenign", "LGGvsGBM", "IDHdetection", "molClass", or "TMBRegression". ds_name is the name of a dataset.
  
  #### Usage example
    To run a CNN model, simply choose your target and subgroup, if any, and type
  ``` shell
  python main.py --dstrain 'TCGA' --target 'IDH' --subgroup 'LGG' --batch=128 --device=1 
  ```

 #### Public models trained on TCGA
We published our models trained on TCGA partitions for each task. 

##### TCGA data portal
https://portal.gdc.cancer.gov/

(note: the models for "cancerVsBenign" was trained BWH dataset. Due to the confidentiality, the associated dataset is not public.)
https://www.dropbox.com/sh/g2c1cl29brrqiul/AAAtB1wbd6txTH99Gw8J_xIXa?dl=0





### Issues
### License 
© [HMS DBMI Yu Lab](https://yulab.hms.harvard.edu/) - This code is made available under the GPLv3 License and is available for non-commercial academic purposes. 

### Acknowledgements
We thank Mr. Alexander Bruce for his assistance with slide scanning at the Digital Imaging Facility, Department of Pathology, Brigham and Women’s Hospital. We thank the Microsoft Azure for Research Award, AWS Cloud Credits for Research, Google Cloud Platform research credit program, the NVIDIA GPU Grant Program, and the Extreme Science and Engineering Discovery Environment (XSEDE) at the Pittsburgh Supercomputing Center (allocation TG-BCS180016) for their computational support. K.-H. Yu is partly supported by the National Institute of General Medical Sciences grant R35GM142879, the Partners’ Innovation Discovery Grant, the Schlager Family Award for Early Stage Digital Health Innovations, and the Blavatnik Center for Computational Biomedicine Award. This work was conducted with support from the Digital Imaging Facility, Department of Pathology, Brigham and Women’s Hospital, Boston, MA and with financial contributions from Brigham and Women’s Hospital.


