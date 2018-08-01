# CultureNet 

CultureNet is a package for building generalized and culturalized deep models to estimate engagement levels from face images of autistic children. 

The baseline generalized deep models (GenNet) are deep convolutional networks composed of layers of the ResNet, a pre-trained deep network, as well as additional network layers. 

The culturalized deep models (CultureNet) follow the same architecture of GenNet; however, after training all layers with joint-culture data, the model freezes network parameters and uses culture-specific and/or child-specific data to fine-tune the last layer of the network. 

## Citation 

If you use this code or these benchmarks in your research, please cite the following publication: Ognjen Rudovic, Yuria Utsumi, Jaeryoung Lee, Javier Hernandez, Eduardo Castelló Ferrer, Björn Schuller, and Rosalind Picard. CultureNet: A Personalized Deep Learning Approach for Engagement Intensity Estimation from Face Images of Children with Autism. 

## Deep Learning Models 

CultureNet consists of seven subject-independent and subject-dependent GenNet and CultureNet models, trained with within-culture, cross-culture, mixed-culture, and joint-culture data, where C0 indicates data from Japan and C1 indicates data from Serbia. 

### Model 1 - Subject Independent, Within-Culture GenNet 
The model is trained and tested on data of children from the same culture. 

### Model 2 - Subject Independent, Cross-Culture GenNet 
The model is trained on children from C0 and tested on children from C1, and vice versa. 

### Model 3 - Subject Independent, Mixed-Culture GenNet 
The model is trained on data randomly selected from 50% of the children from both cultures, then tested on each culture. 

### Model 4 - Subject Independent, Joint-Culture CultureNet 
The joint deep model is trained using training data from both cultures, then the last layer is fine-tuned to each culture separately.  

### Model 5 - Subject Dependent, Within-Culture GenNet 
The model is trained and tested on data of children from the same culture. Training data also includes 20% of target child data. 

### Model 6 - Subject Dependent, Child-Specific GenNet 
The model is trained on 20% of target child data. 

### Model 7 - Subject Dependent, Joint-Culture CultureNet 
The joint deep model is trained using training data from both cultures, then the last layer is fine-tuned to each culture separately, then the last layer is additionally fine-tuned to each target child. 

## Getting Started 

### Installation 

## Running Tests 

## References  
