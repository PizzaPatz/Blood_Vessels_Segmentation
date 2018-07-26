# Blood_Vessels_Segmentation

## About
The experiment on convolutional neural network using [U-Net architecture](https://arxiv.org/abs/1505.04597). The objective of this machine learning project is to train the neural network to recognize blood vessels on retina. Given limited of 20 images to train the model, the segmentation is the best approach to this particular dataset. Segmentation is a way of patching small areas of an image to increase inputs number, this allows the neural network to learn in depth for specific patterns.

## Method
NVIDIA Tesla P100 was used to compute the neurals. Batch size of 32 and learning rate of 0.01 were the setting of the experiment. Computation was done on [xsede](https://www.xsede.org/) and took approximately 10 hours to complete the training.

### Architecture Details
  ![Architecture](https://github.com/PizzaPatz/Blood_Vessels_Segmentation/blob/master/Overview_results/parameters.jpg)
  
### Architecture Overview
  ![Architecture2](https://github.com/PizzaPatz/Blood_Vessels_Segmentation/blob/master/Overview_results/architecture.jpg)

## Results
AUC ROC result is 0.97918092541, which reached state-of-art from [orobix](https://github.com/orobix/retina-unet). However, we have not used mask to reduce computation, thus, our method may gave us a better validation in small epoch size, but we may have used resources more than we should have.

### Accuracy and Loss (50 epochs and 150 epochs)
  ![Acc_Loss](https://github.com/PizzaPatz/Blood_Vessels_Segmentation/blob/master/Overview_results/acc_loss.jpg)

### Prediction
  ![Prediction](https://github.com/PizzaPatz/Blood_Vessels_Segmentation/blob/master/Overview_results/prediction.jpg)
  First Row: Input images
  
  Second Row: Ground truths (Ideal outputs we need)
  
  Third Row: Outputs from training 10 epochs
  
  Fourth Row: Outputs from training 100 epochs
  
  Fifth Row: Outputs from training 150 epochs
  


#### Disclaimer
This is only an experiment, I and my research team have not published an official paper on this experiment yet. If you have any further question regarding this repository, please feel free to email me: patrapee.pongtana@gmail.com
