# Cat-Name-Predictor

## Project Overview
The Cat Image Classifier is a machine learning project designed to classify images of cats based on specific attributes. This project utilizes convolutional neural networks (CNN) and is implemented using Python. The model is trained on a curated dataset described in an Excel file and leverages the pre-trained MobileNetV2 architecture for feature extraction.

## How to Use
1. Ensure you have Python and pip installed.
2. Install all dependencies listed in `requirements.txt`.
3. Run the `app.py` script to start the Flask server for deploying the classifier as a web application.
4. Use the Jupyter notebook `ML_Cats.ipynb` to train the model and explore the training process and evaluations.

## Web Application
To make it easy to use and accessible, the Cat Name Predictor is also deployed as a web application which can be accessed [here](https://miindisponi99-cat-name-predictor-app-brruaa.streamlit.app).

### Testing the Application
To test the application, follow these steps:
1. Visit the [Cat Name Predictor App](https://miindisponi99-cat-name-predictor-app-brruaa.streamlit.app).
2. Upload an image of the cat using the provided interface.
3. Click on the "Predict" button to see the model's prediction.
4. For testing, you can use the image labeled "Prova1" to see how the model performs.

## Repository Structure
- `images`: directory containing images for model training
- `ML_Cats.ipynb`: Jupyter notebook with model training and evaluation
- `CNNEvaluation.py`: script for evaluating the CNN model
- `File_cats.xlsx`: Excel file containing labels and image paths
- `cat_classifier_model.h5`: saved model after training
- `label_encoder_classes.npy`: numpy file to keep track of label encodings
- `app.py`: flask application for model deployment
- `requirements.txt`: list of packages required to run the project
- `.gitignore`: specifies intentionally untracked files to ignore
- `LICENSE`: license file for the project

## Requirements
To install the required Python libraries, run the following command:
```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License.


---

This README provides an overview of the Cat-Name-Predictor repository, including its features, requirements, usage, and detailed descriptions of model deployment using streamlit.
