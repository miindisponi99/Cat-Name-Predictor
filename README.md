# Cat-Name-Predictor

## Project Overview
The Cat Image Classifier is a machine learning project designed to classify images of cats based on specific attributes. This project utilizes convolutional neural networks (CNN) and is implemented using Python. The model is trained on a curated dataset described in an Excel file and leverages the pre-trained MobileNetV2 architecture for feature extraction.

## Repository Structure
├── images                     # Directory containing images for model training
├── .gitignore                 # Specifies intentionally untracked files to ignore
├── CNNEvaluation.py           # Script for evaluating the CNN model
├── File_cats.xlsx             # Excel file containing labels and image paths
├── LICENSE                    # License file for the project
├── ML_Cats.ipynb              # Jupyter notebook with model training and evaluation
├── README.md                  # README file for the project overview
├── app.py                     # Flask application for model deployment
├── cat_classifier_model.h5    # Saved model after training
├── label_encoder_classes.npy  # Numpy file to keep track of label encodings
└── requirements.txt           # List of packages required to run the project
