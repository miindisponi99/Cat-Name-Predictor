import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from keras.utils import to_categorical # type: ignore
from keras.models import Sequential # type: ignore
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input # type: ignore
from keras.preprocessing.image import img_to_array, load_img # type: ignore
from keras.regularizers import l2 # type: ignore
import random

class CatImageClassifier:
    def __init__(self, excel_path, image_dir, image_size=(128, 128)):
        self.excel_path = excel_path
        self.image_dir = image_dir
        self.image_size = image_size
        self.label_encoder = LabelEncoder()
        self.model = None
        self.load_data()
        self.preprocess_data()

    def load_data(self):
        self.data = pd.read_excel(self.excel_path)
        self.images = []
        self.labels = []
        
        for index, row in self.data.iterrows():
            img_path = os.path.join(self.image_dir, row.iloc[0])
            img = load_img(img_path, target_size=self.image_size)
            img_array = img_to_array(img)
            self.images.append(img_array)
            self.labels.append(row.iloc[1])
        
        self.images = np.array(self.images, dtype='float32') / 255.0
        self.labels = np.array(self.labels)
        
    def preprocess_data(self):
        self.labels_encoded = self.label_encoder.fit_transform(self.labels)
        self.labels_categorical = to_categorical(self.labels_encoded)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.images, self.labels_categorical, test_size=0.3, random_state=42)
        
    def build_model(self):
        self.model = Sequential([
            Input(shape=(self.image_size[0], self.image_size[1], 3)),
            Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
            MaxPooling2D((2, 2)),
            Dropout(0.3),
            Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
            MaxPooling2D((2, 2)),
            Dropout(0.3),
            Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
            MaxPooling2D((2, 2)),
            Dropout(0.3),
            Flatten(),
            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.5),
            Dense(len(self.label_encoder.classes_), activation='softmax')
        ])
        
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
    def train_model(self, epochs=500, batch_size=32):
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_data=(self.X_test, self.y_test))
        
    def evaluate_model(self):
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        print(f'Test Accuracy: {accuracy * 100:.2f}%')
        
        y_pred = self.model.predict(self.X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        
        conf_matrix = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=self.label_encoder.classes_, yticklabels=self.label_encoder.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
        
        class_report = classification_report(y_true, y_pred_classes, target_names=self.label_encoder.classes_)
        print(class_report)
        
    def predict_cat(self, image_path):
        img = load_img(image_path, target_size=self.image_size)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = self.model.predict(img_array)
        predicted_label = self.label_encoder.inverse_transform([np.argmax(prediction)])
        return predicted_label[0]
    
    def plot_predictions(self, num_samples=5):
        indices = random.sample(range(len(self.X_test)), num_samples)
        y_true = np.argmax(self.y_test, axis=1)
        y_pred_classes = np.argmax(self.model.predict(self.X_test), axis=1)
        
        for idx in indices:
            img = self.X_test[idx]
            actual_label = self.label_encoder.inverse_transform([y_true[idx]])[0]
            predicted_label = self.label_encoder.inverse_transform([y_pred_classes[idx]])[0]
            
            plt.figure(figsize=(2, 2))
            plt.imshow(img)
            plt.title(f'Actual: {actual_label}\nPredicted: {predicted_label}')
            plt.axis('off')
            plt.show()