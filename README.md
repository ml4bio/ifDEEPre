## ifDEEPre: large protein language-based deep learning enables interpretable and fast predictions of enzyme commission numbers

This is the tensorflow implementation of ifDEEPre. 


## 1. System Requirements

The `ifDEEPre` package is built under the Linux system with the popular softwares [Anaconda](https://www.anaconda.com/) and [Tensorflow](https://www.tensorflow.org/). The versions of the software dependencies that both packages use are provided in the `environment.yml`.

The versions of the software dependencies and data-analysis packages that ifDEEPre has been tested on are given in the `environment.yml`. Users can conveniently create the same environment by running the command:
```
conda env create -f environment.yml
```

The ifDEEPre package does not require any non-standard hardware.


## 2. Installation Guide

### Install the package
The environment that we use is given in `environment.yml`. You can create the same environment by running the command:
```
conda env create -f environment.yml
```


## 3. Demo and Instructions for Using ifDEEPre

You can use the trained ifDEEPre models to predict enzyme commission numbers by navigating to the `./src_v6_Final_server` folder and running the command:
```
python code_4_ifdeepre_inputExample.py
```


## 4. Online Version - ifDEEPre server

The online server of this package is available from this link, [ifDEEPre](https://proj.cse.cuhk.edu.hk/aihlab/ifdeepre/#/), which is freely available without any registration requirement.

Users can directly upload their protein sequences and get accurate enzyme commission number prediction results conveniently after a short time of waiting. Furthermore, the prediction results can be directly downloaded for convenient future usage.






