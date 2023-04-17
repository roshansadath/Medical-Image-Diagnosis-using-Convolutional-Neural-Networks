
# COMP6721-AppliedAI -- Medical Image Diagnosis using Convolutional Neural Networks

## Introduction

Medical Imaging is a critical component of modern healthcare that can aid medical professionals to make more informed diagnostic decisions. Chest X-rays are the most commonly used medical imaging modality and their interpretation can be a time-consuming, challenging, and error-prone process, even for expert radiologists to accurately diagnose diseases. The advancement of AI has triggered a significant transformation in the field of Healthcare, thus, our project aims to utilize state-of-the-art AI models, to assist in the diagnosis of respiratory and cardiovascular conditions in patients. We seek to train advanced Convolutional Neural Networks (CNNs) (AlexNet, ResNet18, and Inceptionv3), with Chest X-Ray images, to compare the performance metrics of these AI models to identify the underlying trends in the data and to diagnose abnormalities accurately.


## Proposed Methodology and Expected Results 

The goal is to evaluate the models based on their metrics and draw insights into their performance on a particular dataset. Since the ResNet18 and Inceptionv3 have deep architectures we expect them to learn more features and perform better than the AlexNet. This can be verified by comparing the Accuracy, Precision, Recall, and F-Score of the models. However, the shallow nature of AlexNet and the lesser number of layers in ResNet18 help them train faster (training time per epoch) than Inceptionv3 and will thus be computationally less expensive.

To perform disease classification from Chest X-rays, we chose the datasets publicly available from Kaggle namely, the Chest Xray Images, Chest X-Ray Dataset for Respiratory Disease Classification, NIH Chest Xray-14.

### Dataset 1

This dataset consists of 5,856 chest X-ray images of patients with and without pneumonia. The data is present in a .jpeg format. It consists of 2 classes Normal (1,342) and Pneumonia (4,514), of size 1024*1024.
Source:  Chest xray images to diagnose pneumonia
— kaggle.com. https://www.kaggle.com/datasets/paulti/chest-xray-images

### Dataset 2

This dataset contains a total of 32,687 images with 5 classes of respiratory diseases, including COVID-19 (4,189), Lung-Opacity (6,012), Normal (10,192), Viral Pneumonia (7,397), and Tuberculosis (4,897). The data is in the form of a .npz file, which is a dictionary containing the image (224*224), image name, and image label. We have sampled 12,000 images from this dataset.

Source: Arkaprabha Basu, Sourav Das, Susmita Ghosh, Sankha
Mullick, Avisek Gupta, and Swagatam Das. Chest
X-Ray Dataset for Respiratory Disease Classification,
2021  -  https://dataverse.harvard.edu/file.xhtml?fileId=5194823&version=5.1 

### Dataset 3

The NIH Chest X-rays dataset is a collection of 112,120 frontal-view chest X-ray images of 30,805 unique patients, of which we sample 20,000 images. The metadata is in a .csv format and the dataset represents 15 different abnormalities diagnosed including Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema,, Emphysema, Fibrosis, Effusion, Pneumonia, No Finding, Pleural thickening, Cardiomegaly, Nodule Mass, and Hernia.

Source: NATIONAL INSTITUTES OF HEALTH CHEST XRAY DATASET. Nih chest x-rays — kaggle.com.
https://www.kaggle.com/datasets/nihchest-xrays/data

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Requirements

To run Python with PyTorch and CUDA, you'll need to have the following requirements installed on your system:

### Python

You'll need to have Python 3.6 or later installed on your system. You can download the latest version of Python from the official website: https://www.python.org/downloads/


### PyTorch

You can install PyTorch using pip. Run the following command in your terminal to install PyTorch (refer - https://pytorch.org/get-started/locally/): 
    `pip install torch torchvision`


### CUDA

To use PyTorch with CUDA, you'll need to have a compatible NVIDIA GPU and the CUDA toolkit installed on your system. You can download the CUDA toolkit from the official NVIDIA website: https://developer.nvidia.com/cuda-downloads


Make sure to select the correct version of the CUDA toolkit that's compatible with your GPU and operating system.

Once you have the CUDA toolkit installed, you'll need to set the following environment variables:

  `export PATH=/usr/local/cuda/bin:$PATH`
  `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`
 
You can add these environment variables to your `.bashrc` file to make them persistent.

### cuDNN

You'll also need to install the cuDNN library, which is a CUDA-accelerated library for deep neural networks. You can download the cuDNN library from the NVIDIA website: https://developer.nvidia.com/cudnn


Make sure to select the version of the cuDNN library that's compatible with your CUDA toolkit and operating system.

Once you have all of these requirements installed, you should be able to run Python with PyTorch and CUDA on your system.



### ALTERNTIVE - USE Google Colab 
### Using PyTorch with CUDA on Google Colab

To use Python with PyTorch and CUDA on Google Colab, you can follow these steps:

1. Open a new notebook in Google Colab.
2. Go to "Runtime" and select "Change runtime type".
3. Under "Hardware accelerator", select "GPU" and click "SAVE".
4. Install PyTorch using pip. Run the following command in a code cell: `!pip install torch torchvision`
5. Verify that PyTorch is installed by importing it and printing the version number:

`python
import torch
print(torch.__version__)`
This should output the version number of PyTorch.

6. To use CUDA, you don't need to install anything additional on Google Colab since it already has the CUDA toolkit and drivers installed on its virtual machines.
7. You can now start using PyTorch with CUDA in your Google Colab notebook.



## Training and Validating Our PyTorch Models in Google Colab

To train and validate our PyTorch models in Google Colab, you can follow these steps:
1. Load the dataset into a format that PyTorch can use, such as a `torch.utils.data.Dataset`.
2. Split the dataset into training and validation sets, for example using `torch.utils.data.random_split()`.
3. Define a loss function, for example using `torch.nn.CrossEntropyLoss()`.
5. Define an optimizer, for example using `torch.optim.Adam()`.
6. Train the model for a specified number of epochs or until convergence.
7. Validate the model using validation split.
8. Save the trained model to disk using `torch.save()`.

##### Code for all the above steps is available in folders:
    COMP6721_Winter2023_GroupN/Simple Chest XRay/
    COMP6721_Winter2023_GroupN/Harvard Chest XRay/
    COMP6721_Winter2023_GroupN/Hyperparameter Tuning/
    COMP6721_Winter2023_GroupN/Transfer Learning/
    COMP6721_Winter2023_GroupN/NIH Chest XRay/
    
all the utility functions used can be found here : COMP6721_Winter2023_GroupN/utils.ipynb

The links for original full datasets can be found in Introduction section above.

##### Here is a link for a sample dataset of 100 images - https://drive.google.com/drive/folders/10e-Yf_PxUTCDvh97mU1C5dp5tz7flEKk?usp=sharing


## Running a Pre-Trained PyTorch Model on a Sample Test Dataset

To run a pre-trained PyTorch model on a provided sample test dataset, you can follow these steps:

1. Obtain the pre-trained model file as a `.pth` file, from the folders mentioned below
2. Download or obtain the sample test dataset from here - https://drive.google.com/drive/folders/10e-Yf_PxUTCDvh97mU1C5dp5tz7flEKk?usp=sharing.
3. Create a PyTorch model using the same architecture and input format as the pre-trained model, and load the pre-trained parameters using `torch.load()`.
4. Apply the model to the sample test dataset using a data loader, for example using `torch.utils.data.DataLoader`.
5. Compute the model's output on each sample in the test dataset, for example using the `model()` function.

##### example code that loads model from weights:
```
python
import torch
from torch.utils.data import DataLoader, Dataset
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
model.load_state_dict(torch.load('your_model_weights.pth'))
```

##### Saved Model files (.pth) are found in below links
    ResNet18 - Dataset 1 : https://drive.google.com/file/d/1-AJMU30_2ppw-yORpCqfM-zvI_JgaarB/view?usp=sharing.
    AlexNet - Dataset 2 : https://drive.google.com/file/d/1FEsD9htWVHe7BxBNtFT2aWeT1GhzAYyv/view?usp=share_link.
    Inceptionv3 - Dataset 2 : https://drive.google.com/file/d/1-9kRtWVIPHfzm5V3vXFN4H-qaBO9tpul/view?usp=sharing.
    ResNet18 - Dataset 3 : https://drive.google.com/file/d/176zEWSmk3QoX_bdImW8MeeHhYFn-O_Xm/view?usp=sharing. 

##### the code to run above steps can be found in the following directories in this repo:

    COMP6721_Winter2023_GroupN/Simple Chest XRay/
    COMP6721_Winter2023_GroupN/Harvard Chest XRay/
    COMP6721_Winter2023_GroupN/Hyperparameter Tuning/
    COMP6721_Winter2023_GroupN/Transfer Learning/
    COMP6721_Winter2023_GroupN/NIH Chest XRay/





### Sample data

Follow the drive link - https://drive.google.com/drive/folders/10e-Yf_PxUTCDvh97mU1C5dp5tz7flEKk?usp=sharing
download the whole folder and place it in the same directory where the we run the code to train and test




### Ablation results

hyper parameter tuning on dataset 2

| optimizer     |  Adam  |        |        |         |   SGD  |        |        |        |
|---------------|:------:|:------:|:------:|:-------:|:------:|:------:|:------:|:------:|
| learning rate |  0.005 | 0.0001 | 0.0005 | 0.00001 |  0.05  |  0.01  |  0.005 |  0.001 |
|               |        |        |        |         |        |        |        |        |
| Accuracy      | 0.3164 | 0.8539 | 0.8076 |  0.7112 | 0.1335 | 0.8034 | 0.7640 | 0.6477 |
| Precision     | 0.0633 | 0.8600 | 0.7991 |  0.6976 | 0.0267 | 0.8052 | 0.7983 | 0.6610 |
| Recall        | 0.2000 | 0.8359 | 0.7972 |  0.6884 | 0.2000 | 0.7843 | 0.7945 | 0.6011 |
| F-Score       | 0.0961 | 0.8462 | 0.7960 |  0.6915 | 0.0471 | 0.7925 | 0.7937 | 0.6095 |
|               |        |        |        |         |        |        |        |        |







