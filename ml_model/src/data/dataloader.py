import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def load_images_from_folder(folder):
    images = []
    labels = []
    for subfolder in os.listdir(folder):
        label = subfolder
        path = os.path.join(folder, subfolder)
        for filename in os.listdir(path):
            img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(label)
    return images, labels

def load_data(dataset_dir, test_size=0.2, random_state=42):
    # Load all data
    X, y = load_images_from_folder(dataset_dir)
    X = np.array(X)
    y = np.array(y)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_train = to_categorical(y_train, 2)
    y_test = label_encoder.transform(y_test)
    y_test = to_categorical(y_test, 2)

    return X_train, y_train, X_test, y_test, label_encoder