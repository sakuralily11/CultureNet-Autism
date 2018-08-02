# CultureNet 

CultureNet is a package for building generalized and culturalized deep models to estimate engagement levels from face images of autistic children. 

The baseline generalized deep models (GenNet) are deep convolutional networks composed of layers of the ResNet, a pre-trained deep network, as well as additional network layers. 

The culturalized deep models (CultureNet) follow the same architecture of GenNet; however, after training all layers with joint-culture data, the model freezes network parameters and uses culture-specific and/or child-specific data to fine-tune the last layer of the network. 

Face image data are unidentifiable facial features, obtained in the output of the fine-tuned residual network (ResNet), a deep network optimized for object classification. 

## Citation 

If you use this code or these benchmarks in your research, **please cite the following publications**: 

Ognjen Rudovic, Jaeryoung Lee, Lea Mascarell-Maricic, Björn Schuller, and Rosalind Picard. Measuring Engagement in Robot-Assisted Autism Therapy: A Cross-Cultural Study. 

Available via [Frontiers in Robotics and AI](https://doi.org/10.3389/frobt.2017.00036).

Ognjen Rudovic, Yuria Utsumi, Jaeryoung Lee, Javier Hernandez, Eduardo Castelló Ferrer, Björn Schuller, and Rosalind Picard. CultureNet: A Personalized Deep Learning Approach for Engagement Intensity Estimation from Face Images of Children with Autism. 

## Deep Learning Models 

CultureNet consists of seven subject-independent and subject-dependent GenNet and CultureNet models, trained with within-culture, cross-culture, mixed-culture, and joint-culture data, where C0 indicates data from Japan and C1 indicates data from Serbia. 

For all subject-independent models, we used a leave-one-child-out evaluation, where we performed training on 80% of each training child's data, validation on the remaining 20% of each training child's data, and testing on 80% of the target child's data. 

For all subject-dependent models, we performed training on 20% of each training child's data, validation on 20% of each validation child's data, and testing on 80% of the target child's data. 

All experiments were repeated 10 times, each time starting from a different random initialization of the deep models. 

### Model 1 - Subject Independent, Within-Culture GenNet 
The model is trained and tested on data of children from the same culture. 

### Model 2 - Subject Independent, Cross-Culture GenNet 
The model is trained on data of children from C0 and tested on data of children from C1, and vice versa. 

### Model 3 - Subject Independent, Mixed-Culture GenNet 
The model is trained on data from both cultures, then tested on each culture. Note: This is the first step of model 4. 

### Model 4 - Subject Independent, Joint-Culture CultureNet 
The joint deep model is trained on data from both cultures, then the last layer is fine-tuned to each culture separately. Note: This is the first two steps of model 7. 

### Model 5 - Subject Dependent, Within-Culture GenNet 
The model is trained and tested on data of children from the same culture. Training data also includes 20% of target child data. Note: This is the subject-dependent version of model 1. 

### Model 6 - Subject Dependent, Child-Specific GenNet 
The model is trained on 20% of target child data. 

### Model 7 - Subject Dependent, Joint-Culture CultureNet 
The joint deep model is trained on data from both cultures, then the last layer is fine-tuned to each culture separately, then the last layer is additionally fine-tuned to each target child. 

## Getting Started 

CultureNet allows for prediction of engagement levels from facial features of autistic children, obtained in the output of ResNet. 

### Installation 

To install CultureNet, clone the repository:

```
$ git clone https://github.com/yuriautsumi/CultureNet-Autism.git 
```

### Requirements 

We do not provide the face image data itself. To obtain the data, please email yutsumi@mit.edu. 

The requirements can be installed via [pip](https://pypi.python.org/pypi/pip) as follows (use ```sudo``` if necessary): 

```
$ pip install -r requirements.txt 
```

To demo, run the ```runIROS.py``` script. 
