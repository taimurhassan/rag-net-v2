# Clinically Verified Hybrid Deep Learning System for Retinal Ganglion Cells Aware Grading of Glaucomatous Progression

## Introduction
<p align="justify">
We present a novel strategy encompassing a hybrid convolutional network RAG-Net<sub>v2</sub> that extracts the retinal nerve fiber layer (RNFL), ganglion cell with the inner plexiform layer (GC-IPL) and GCC regions, allowing thus a quantitative screening and grading of glaucomatous subjects. Furthermore, RAG-Net has additionally been tested to screen the glaucomatous cases through retinal fundus images. The block diagram of the proposed framework is presented below:
</p>

![RAG-Netv2](/images/Picture10.png) 
<p align="center"> Block Diagram of the Proposed Framework</p>

This repository contains the source code of our paper currently under review in IEEE Transactions on Biomedical Engineering. The proposed framework is developed using <b>TensorFlow 2.3.0</b> and <b>Keras APIs</b> with <b>Python 3.8.5</b>. Moreover, the results are compiled through <b>MATLAB R2020a</b>. The detailed steps for installing and running the code are presented below:

## Installation
To run the codebase, following libraries are required. Although, the framework is developed using Anaconda. But it should be compatable with other platforms.

1) TensorFlow 2.3.0 
2) Keras 2.0.0 or above
3) OpenCV 4.2
4) Imgaug 0.2.9 or above
5) Tqdm
6) Matplotlib

Alternatively, we also provide a yml file that contains all of these packages.

## Dataset
The proposed framework has been tested on the following public datasets:

1) Armed Forces Institute of Ophthalmology (AFIO) Dataset [URL](https://www.sciencedirect.com/science/article/pii/S2352340920302365) (contains correlated fundus and OCT scans of glaucomatous and healthy subjects)
2) ORIGA Dataset [URL](https://drive.google.com/drive/folders/1VPCvVsPgrfPNIl932xgU3XC_WFLUsXJR) (contains fundus images of healthy and glaucomic patients)

## Steps 
<p align="justify">
<b>For OCT Analysis</b>

1) Download the AFIO dataset
2) Use 'preprocessor.m' or 'structure_tensor_get.py' to preprocess the input scans
3) Use 'augmentation.py' or 'augmentor.m' to augment the training scans
4) Put the augmented training images in '…\trainingDataset\train_images' and '…\codebase\models\trainingSet' folders. The former one is used for segmentation and the latter one is used for the classification purposes.
5) Put the training annotations (for segmentation) in '…\trainingDataset\train_annotations' folder
6) Put validation images in '…\trainingDataset\val_images' and '…\codebase\models\validationSet' folders. The former one is used for segmentation and the latter one is used for the classification purposes.
7) Put validation annotations (for segmentation) in '…\trainingDataset\val_annotations' folder. Note: the images and annotations should have same name and extension (preferably .png).
8) Put test images in '…\testingDataset\test_images' folder and their annotations in '…\testingDataset\test_annotations' folder
9) Use 'trainer.py' file to train RAG-Net<sub>v2</sub> on preprocessed scans and also to evaluate the trained model on test scans. The results on the test scans are saved in ‘…\testingDataset\segmentation_results’ folder. This script also saves the trained model in 'model.h5' file.
10) Run 'ragClassifier.py' script to classify the preprocessed test scans as normal or glaucomic. The results are saved as a mat file in '…\testingDataset\diagnosisResults' folder. Note: step 10 can only be done once the step 9 is finished because the model trained in step 9 is required in step 10. 
11) Once step 10 is completed, run 'trainSVM.m' script to train the SVM model for grading the severity of the classified glaucomic scans.
12) Once the SVM is trained, run 'glaucomaGrader.m' to get the grading results.
13) The trained models can also be ported to MATLAB using ‘kerasConverter.m’ (this step is optional and only designed to facilitate MATLAB users if they want to avoid python analysis).
14) Some additional results (both qualitative and quantitative) of the proposed framework are also presented in the '…\results' folder. 

<b>For Fundus Analysis</b>

15) Download the desired dataset
16) Put the training scans in '…\codebase\models\trainingSet\glaucoma' and '…\codebase\models\trainingSet\normal' folders
17) Put the validation scans in '…\codebase\models\validationSet\glaucoma' and '…\codebase\models\validationSet\normal' folders
18) Put the test scans in '…\testingDataset\fundus_test_images' folder.
19) Uncomment the path at line 44 within the 'ragClassifier.py' file. Note: if you want to perform OCT analysis again, this line has to be commented again
20) Run 'ragClassifier.py' to produce classify normal and glaucomic fundus scans. The results will saved in a mat file within  '…\testingDataset\diagnosisResults' folder once the analysis is completed. Note: Before running 'ragClassifier.py', please make sure you have the saved 'model.h5' file generated through step 9 because RAG-Net<sub>v2</sub> classification model initially adapts the weights of the trained RAG-Net<sub>v2</sub> segmentation unit for faster convergence.
</p>

## Results
We have provided both the quantitative and qualitative results in the 'results' folder. Please contact us if you want to get the trained model instances.

## Citation
If you use RAG-Net<sub>v2</sub> (or any part of this code in your research), please cite the following paper:

```
@article{ragnetv2,
  title   = {Clinically Verified Hybrid Deep Learning System for Retinal Ganglion Cells Aware Grading of Glaucomatous Progression},
  author  = {Hina Raja and Taimur Hassan and Muhammad Usman Akram and Naoufel Werghi},
  journal = {Submitted in IEEE Transactions on Biomedical Engineering},
  year = {2020}
}
```

## Contact
If you have any query, please feel free to contact us at: taimur.hassan@ku.ac.ae.
