# Emotion Classification with Convolutional Neural Networks (CNNs)

## Overview:
- Facial expressions are one of the ways humans communicate. Deep learning approaches in human and computer interactions are used in artificial intelligence research as an effective system application process. The detection of emotions or facial expressions in psychology necessitates the examination and evaluation of decisions in anticipating a person's feelings or a group of people communicating. This study proposes the development of a system that can predict and recognize the classification of facial emotions using the Convolution Neural Network (CNN) algorithm and feature extraction.

- Data preprocessing, facial feature extraction, and facial emotion classification are the three key steps in the notebook. Facial expressions were predicted with the accuracy of 62.66 percent with using the Convolutional Neural Network (CNN). This algorithm was evaluated on a publicly available dataset from the FER2013 database, which has 35887 48x48 grayscale face images each representing one of the emotions.
## Prerequisites

- A Google account to access [Google Colab](https://colab.research.google.com/).
- A Kaggle account for [dataset](https://www.kaggle.com/) download.

## Setup

1. **Google Colab Initialization**:
    - Open Google Colab and create a new Python3 notebook.
    - Set your runtime type to GPU for faster computation through `Runtime > Change runtime type`.

2. **Kaggle Dataset Download**:
         Download your desired dataset from [kaggel](https://www.kaggle.com/datasets/deadskull7/fer2013) You can either:
    - Upload it directly to Google Colab using the upload button in the sidebar.
    - Save the dataset to Google Drive and access it from Colab using `drive.mount('/content/drive')`.
## Key Features:
- **Data Preprocessing**: The input images are processed and augmented for better generalization.
- **Modeling**: A CNN architecture tailored for facial emotion recognition.
- **Early Stopping and Model Checkpointing**: Implement early stopping based on validation accuracy to prevent overfitting and save the best-performing model.
- **Performance Visualization**: Plot training and validation losses and accuracies to monitor model convergence and performance.
- **Fine-Tuning**: Experiment with different hyperparameters and optimization strategies to enhance the model's performance.
- **Evaluation Metrics**: Utilize a confusion matrix and a detailed classification report to assess the model's performance on individual emotion classes.

## Results:
The initial model trained using the Adam optimizer achieved a test accuracy of approximately 59.2%. We then attempted to fine-tune the model by switching to the SGD optimizer, but the performance dropped significantly. Further fine-tuning with the Adam optimizer, by adjusting the learning rate and epochs, resulted in an improved test accuracy of approximately 64.3%.

## Tools and Libraries Used:
- TensorFlow & Keras
- Matplotlib for visualization
- Seaborn for advanced visualizations, especially the heatmap
- Scikit-learn for metrics and evaluations

## Future Improvements:
- Investigate more advanced CNN architectures like ResNet or VGG.
- Implement transfer learning by leveraging pre-trained models.
- Address class imbalance in the dataset with techniques like oversampling, undersampling, or synthetic data generation.
  
