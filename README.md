# Seminar in Business Analytics & Quantitative Marketing 2021. Team 1

Created by: Vladyslav Matsiiako (476414), Anouk Veltman (466547), Luca Zampierin (454696), & Laura Zwiers (468008)

This is the repository in which we store all our codes and data sets used during the seminar. 

The goal of this project is to find out how PCA and autoencoders compare in the dimensionality reduction task. As our domain, we take a MNIST data set of American Sign Language. We use PCA, Deep Autoencoders and Denoising Deep Autoencoders and compare them in the ability to retain the structure of the original data and in their clustering performance. We also study the clustring properties of the data using t-SNE. Following this, all the sections of our repository are explained. 

### 1. Autoencoders
- `main.py`
  
  The file where we run the cross-validation, perform training, and evaluate the performance of the obtained models. The evaluation is done using the reconstruction of original pictures, comparing the MSEs between training and validation sets, and visualizing the features detected by the first layer of the Autoencoder models. Additionally, in this file, we store the reduced datasets for future clustering analysis (in the `Data` folder).
- `Deep_Autoencoder_model.py`

  Here, we initialize the classes for `DeepAutoencoder` and `DenoisingDeepAutoencoder`. In both classes, we combine pretraining and finetuning. The difference of the denoising class is that we add extra noise in the fine-tuning part. All the computation are performed inside the .fit method. The necessary inputs are pre-training noise type, pre-training noise parameter, fine-tuning noise type*, fine-tuning noise parameter*, batch size, hidden layers, training set, validation set, number of epochs of fine-tuning, number of epochs of pre-training, learning rate in the pre-training stage, learning rate in the fine-tuning stage.
  
  <sup>\* parameters that are relevant only to Denoising Deep Autoencoder classes.</sup>
- `pretraining_DAE.py`

  Initializes the `PretrainingDAE` class. This class is used to represent the Single Layer Denoising Autoencoder used for pretraining. 
- `finetuning_AE.py`

  Initializes the `FinetuningAE` class. This class is used to represent the Deep Autoencoder used for finetuning constructed using a list of stacked denoising autoencoder models.
- `utils.py`

  In this file, we store the functions that are used in the evaluation steps of `main.py`.
### 2. Clustering
- `kmeans.py`

  In this file, we analyze the clustering performance of k-means on the original data as well as the data sets obtained from PCA, Deep Autoencoder, and Denoising Deep Autoencoder models. Here, we use the Homegeneity, Completeness, and V-measure scores.
- `kmeansreduced.py`

  This is the file in which perform the clustering analysis of the reduced data set (omitting the pictures with fist-like signs).
- `kmedoids.py`

  Here, we analyze the clustering performance of k-medoids on the original data as well as the data sets obtained from PCA, Deep Autoencoder, and Denoising Deep Autoencoder models. Again, we use the Homegeneity, Completeness, and V-measure scores.
  
### 3. Data

- `sign_mnist_train.csv` is the data set containing all the train data in the original format. 
- `sign_mnist_test.csv` is the data set containing all the test data in the original format.
- `Final_train_denoising_ae.csv` is the data set containing the reduced train data obtained from the Denoising Deep Autoencoder model.
- `Final_test_denoising_ae.csv` is the data set containing the reduced test data obtained from the Denoising Deep Autoencoder model.
- `Final_train_ae.csv` is the data set containing the reduced train data obtained from the Deep Autoencoder model.
- `Final_test_ae.csv` is the data set containing the reduced test data obtained from the Deep Autoencoder model.
- `exploratory_data_analysis.py`
- `preprocessing.py`

  In this file, we preprocess the data sets which will be used after and save them as `.csv` files. 

### 4. Intrinsic Dimensionality 

- `ID_estimation.py` 

  In this file, we estimate the intrinsic dimensionality based on our train data. The two methods used are Correlation Dimension and an algorithm based on eigenvalues.
  
### 5. Measures

- `Trustworthiness_continuity.py`

  Here, we perform the calculations of the Trustworthiness and Continuity measures for all the available data sets.
  
### 6. PCA

- `pca.py`

  
- `pca_reconstruction.py`
