## ifDEEPre: large protein language-based deep learning enables interpretable and fast predictions of enzyme commission numbers

![image](https://github.com/ml4bio/ifDEEPre/assets/16831407/cd6aedec-db24-49f2-a93b-29a7e6fd4cfb)
<p align="center">
    <em>Figure 1: Architecture of ifDEEPre for accurate enzyme predictions.</em>
</p>
This is the tensorflow implementation of ifDEEPre. 

## 1. System Requirements

The `ifDEEPre` package is built under the Linux system with the popular software:
- [Anaconda](https://www.anaconda.com/) 
- [Tensorflow](https://www.tensorflow.org/) >= 1.9
- Pytorch>= 1.10.2
- numpy>=1.19.2
  
The versions of the software dependencies that all packages use are provided in the `environment.yml`.


## 2. Installation Guide

### Build the the conda environment an install packages
The versions of the software dependencies and data-analysis packages that ifDEEPre has been tested on are given in the `environment.yml`. The ifDEEPre package does not require any non-standard hardware. You can create the same environment by running the command:

```
git clone https://github.com/ml4bio/ifDEEPre.git
cd ifDEEPre
conda env create -f environment.yml
```
After successful installation without errors, you can activate the conda environment:
```
conda activate tf36_ifdeepre
```
### Download pre-trained model
Download the database and trained models from [Google Drive](https://drive.google.com/drive/folders/1qEMzaDas9M0PaZHrHWNcMby82Ef_Qun1). Save the corresponding files in the ```Database``` folder.

## 3. Demo and Instructions for Using ifDEEPre

You can use the trained ifDEEPre models to predict enzyme commission numbers by navigating to the `./src_v6_Final_server` folder and running the command:
```
cd ./src_v6_Final_server
python code_4_ifdeepre_inputExample.py
```


## 4. Online Version - ifDEEPre server

You are strongly encouraged to use the online server if you do not want to get your hands dirty. The online server is available on our lab's website: [ifDEEPre](https://proj.cse.cuhk.edu.hk/aihlab/ifdeepre/#/), which is free to use without any registration requirement.


Users can directly upload their protein sequences and get accurate enzyme commission number prediction results conveniently after a short time of waiting. Furthermore, the prediction results can be directly downloaded for convenient future usage.
<img width="667" alt="edfb05503195f57c0dd5a8dc94507b0" src="https://github.com/ml4bio/ifDEEPre/assets/16831407/ad2d43b4-0500-4486-a8dc-a4f2fcfa04fa">





