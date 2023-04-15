
# COMP6721-AppliedAI -- Medical Image Diagnosis using Convolutional Neural Networks

## Introduction

Medical Imaging is a critical component of modern healthcare that can aid medical professionals to make more informed diagnostic decisions. Chest X-rays are the most commonly used medical imaging modality and their interpretation can be a time-consuming, challenging, and error-prone process, even for expert radiologists to accurately diagnose diseases. The advancement of AI has triggered a significant transformation in the field of Healthcare, thus, our project aims to utilize state-of-the-art AI models, to assist in the diagnosis of respiratory and cardiovascular conditions in patients. We seek to train advanced Convolutional Neural Networks (CNNs) (AlexNet, ResNet18, and Inceptionv3), with Chest X-Ray images, to compare the performance metrics of these AI models to identify the underlying trends in the data and to diagnose abnormalities accurately.


## Proposed Methodology and Expected Results 

The goal is to evaluate the models based on their metrics and draw insights into their performance on a particular dataset. Since the ResNet18 and Inceptionv3 have deep architectures we expect them to learn more features and perform better than the AlexNet. This can be verified by comparing the Accuracy, Precision, Recall, and F-Score of the models. However, the shallow nature of AlexNet and the lesser number of layers in ResNet18 help them train faster (training time per epoch) than Inceptionv3 and will thus be computationally less expensive.

To perform disease classification from Chest X-rays, we chose the datasets publicly available from Kaggle namely, the Chest Xray Images, Chest X-Ray Dataset for Respiratory Disease Classification, NIH Chest Xray-14.

### Dataset 1

This dataset consists of 5,856 chest X-ray images of patients with and without pneumonia. The data is present in a .jpeg format. It consists of 2 classes Normal (1,342) and Pneumonia (4,514), of size 1024*1024 [7].

#### Source : https://www.kaggle.com/datasets/paulti/chest-xray-images

### Dataset 2

This dataset contains a total of 32,687 images with 5 classes of respiratory diseases, including COVID-19 (4,189), Lung-Opacity (6,012), Normal (10,192), Viral Pneumonia (7,397), and Tuberculosis (4,897). The data is in the form of a .npz file, which is a dictionary containing the image (224*224), image name, and image label. We have sampled 12,000 images from this dataset.

#### Source : 
  Arkaprabha Basu, Sourav Das, Susmita Ghosh,
  Sankha Mullick, Avisek Gupta, and Swagatam Das.
  Chest X-Ray Dataset for Respiratory Disease Classification,2021.

### Dataset 3

The NIH Chest X-rays dataset is a collection of 112,120 frontal-view chest X-ray images of 30,805 unique patients, of which we sample 20,000 images. The metadata is in a .csv format and the dataset represents 15 different abnormalities diagnosed including Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, Emphysema, Fibrosis,Effusion, Pneumonia, No Finding, Pleural thickening, Cardiomegaly, Nodule Mass, and Hernia. 

#### Source :  https://www.kaggle.com/datasets/nih-chest-xrays/data
