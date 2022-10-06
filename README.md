# Brain MRI Image Classification Using Deep Learning Models  
[![Made With Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=black)](https://www.python.org/) [![Tensorflow-Badge](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/) [![Keras-Badge](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io/) [![Jupyter-Notebook-Badge](https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white)](https://nbviewer.jupyter.org/github/strikersps/Brain-MRI-Image-Classification-Using-Deep-Learning/blob/main/Brain-Tumor-MRI-Image-Classification.ipynb) [![NVIDIA-Tesla-T4-Badge](https://img.shields.io/badge/NVIDIA-TeslaT4-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://colab.research.google.com/github/d2l-ai/d2l-tvm-colab/blob/master/chapter_gpu_schedules/arch.ipynb) ![Repo-Size-Badge](https://img.shields.io/github/repo-size/strikersps/Brain-MRI-Image-Classification-Using-Deep-Learning?color=%23ff0000&style=for-the-badge) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Nm8JLCptOMqFHRtnXZeux_qoW9wtAgD_?usp=sharing)
------------------------------------------------------------------------------------------------------------------------------------------------------------------
[![GitHub-Contributors](https://img.shields.io/github/contributors/strikersps/Brain-MRI-Image-Classification-Using-Deep-Learning.svg)](https://github.com/strikersps/Brain-MRI-Image-Classification-Using-Deep-Learning/graphs/contributors) [![GitHub-Issues](https://img.shields.io/github/issues/strikersps/Brain-MRI-Image-Classification-Using-Deep-Learning?style=flat-square)](https://github.com/strikersps/Brain-MRI-Image-Classification-Using-Deep-Learning/issues) [![GitHub-Stars](https://img.shields.io/github/stars/strikersps/Brain-MRI-Image-Classification-Using-Deep-Learning?style=flat-square)](https://github.com/strikersps/Brain-MRI-Image-Classification-Using-Deep-Learning/stargazers) [![GitHub-Forks](https://img.shields.io/github/forks/strikersps/Brain-MRI-Image-Classification-Using-Deep-Learning?style=flat-square)](https://github.com/strikersps/Brain-MRI-Image-Classification-Using-Deep-Learning/network/members)  

![Brain-MRI-Image-Classification-Using-Deep-Learning-Cover-Photo](https://github.com/strikersps/Brain-MRI-Image-Classification-Using-Deep-Learning/blob/main/Project-Cover-Photo.jpg)  

## Introduction  
- The occurrence of brain tumor patients in India is steadily rising, more and more cases of brain tumors are reported each year in India across varied age groups. The International Association of Cancer Registries (IARC) reported that there are over 28,000 cases of brain tumours reported in India each year and more than 24,000 people reportedly die due to brain tumours i.e **85.7%** people die annually from the total reported cases. Brain tumors are a serious condition and in most cases fatal in later stages if not detected early on.

- Healthcare sector can benefit significantly from the field of Artificial Intelligence by developing systems which have the capability to detect these fatal diseases in the early stages because most diseases when detected early can be treated successfully before it's too late and same is the case with various different kinds of cancer.

- **The goal of the project was to develop a deep learning model which has the capability to classify the brain MRI images consisting of tumors with higher accuracy.**

## How to Run  
- To view the jupyter notebook, click on the badge: [![Jupyter-Notebook-Badge](https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white)](https://nbviewer.jupyter.org/github/strikersps/Brain-MRI-Image-Classification-Using-Deep-Learning/blob/main/Brain-Tumor-MRI-Image-Classification.ipynb)  

- To view and execute the notebook in Google Colab, click on the badge: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Nm8JLCptOMqFHRtnXZeux_qoW9wtAgD_?usp=sharing)

- Before executing the Google Colab notebook make sure you can access the dataset from your Google Colab notebook for which you can create the shortcut to the Google Drive link: https://drive.google.com/drive/folders/11QIC82FBdAyq0PUwLVNd22i-oq6lcat1?usp=sharing on your google drive and then set the path to this shortcut by changing the value of `DATA_PATH_DIR` variable in the [notebook](https://colab.research.google.com/drive/1Nm8JLCptOMqFHRtnXZeux_qoW9wtAgD_#scrollTo=_Ta1TX_tMUQl) to get access to the datasets after you have given access to your google drive by executing the following piece of code:
  ```python3
  from goolge.colab import drive
  drive.mount("/content/gdrive/")
  ```
```diff
- CAUTION: The total size of repository is atleast 2.5 GB. 
- Make sure you have enough space before cloning.
```
## About Dataset  
- Refer to [README.md](https://github.com/strikersps/Brain-MRI-Image-Classification-Using-Deep-Learning/blob/main/Brain-Tumor-Dataset/README.md) file in the [Brain Tumor Dataset directory](https://github.com/strikersps/Brain-MRI-Image-Classification-Using-Deep-Learning/tree/main/Brain-Tumor-Dataset) in this repository to get a clear idea about the dataset and the preprocessing steps.  
- The below image gives a glimpse about the different kinds of tumors with its localisation through a binary map after pre-processing the `.mat` file in which the image data was stored.  
![Brain-MRI-Images-With-Localisation-Masks](https://github.com/strikersps/Brain-MRI-Image-Classification-Using-Deep-Learning/blob/main/Brain-Tumor-MRI-With-Localisation-Masks.png)    

## Results  
- Developed 3 Deep Neural Network models i.e. Multi-Layer Perceptron, AlexNet-CNN, and Inception-V3 in order to classify the Brain MRI Images to 4 different independent classes.  
- Inception-V3 model used is a pre-trained on the ImageNet dataset which consist of 1K classes but for this project we have tuned the later part i.e. the Fully-Connected part of the model while retaining the weights of the CNN part to satisfy the needs of this work. 
- Below table provides the results obtained on the testing dataset: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Nm8JLCptOMqFHRtnXZeux_qoW9wtAgD_#scrollTo=nobbz__oKJgb)
![Model-Results-On-Testing-Dataset](https://github.com/strikersps/Brain-MRI-Image-Classification-Using-Deep-Learning/blob/main/Model-Results-On-Testing-Dataset.png)
- **The pre-trained Inception-V3 model has performed significantly well with an accuracy of `82.57%` as compare to AlexNet-CNN and Multi-Layer Perceptron deep neural network model.**  

## Future Works  
- To improve the robustness and accuracy of model further we can develop a efficient Data-Augmentation pipline in order to expose the CNN model to more variants of the Brain MRI Images.  
- Training process can be migrated to TPUs (Tensor Processing Units) by representing the data in TFRecord format for significant reduction in training time.  
- Implementation of [Region Convolutional Neural Networks (R-CNN)](https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e) to not only detect the tumor in a Brain MRI Image but also label, localise and highlight the tumor region.

##### NOTE: If you want to cite this repository, then please copy the respective style information (APA or BibTex) provided under `cite this repository` option as shown in the tutorial: https://github.blog/wp-content/uploads/2021/08/GitHub-citation-demo.gif

###### GNU General Public License v3.0  
[![GPL-V3-Badge](https://img.shields.io/github/license/strikersps/Brain-MRI-Image-Classification-Using-Deep-Learning?color=red&style=for-the-badge)](https://github.com/strikersps/Brain-MRI-Image-Classification-Using-Deep-Learning/blob/main/LICENSE)  
