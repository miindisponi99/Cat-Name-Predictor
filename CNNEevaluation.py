import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img # type: ignore
from keras.applications import MobileNetV2 # type: ignore

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
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(self.image_size[0], self.image_size[1], 3))
        x = base_model.output
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(len(self.label_encoder.classes_), activation='softmax')(x)
        self.model = Model(inputs=base_model.input, outputs=predictions)
        for layer in base_model.layers:
            layer.trainable = False
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def create_datagen(self):
        return ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

    def train_model(self, epochs=100, batch_size=16):
        datagen = self.create_datagen()
        class_weights = self.get_class_weights()
        self.model.fit(
            datagen.flow(self.X_train, self.y_train, batch_size=batch_size),
            steps_per_epoch=len(self.X_train) // batch_size,
            epochs=epochs,
            validation_data=(self.X_test, self.y_test),
            class_weight=class_weights
        )

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

    def get_class_weights(self):
        class_weights = compute_class_weight(
            class_weight='balanced', 
            classes=np.unique(self.labels_encoded), 
            y=self.labels_encoded
        )
        return dict(enumerate(class_weights))

    def plot_class_distribution(self):
        plt.figure(figsize=(10, 6))
        sns.countplot(x=self.labels)
        plt.title('Class Distribution')
        plt.show()

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