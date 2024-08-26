import numpy as np

def preprocess_data(X_train, X_test=None):
    # Normalize the images
    X_train = X_train / 255.0
    if X_test is not None:
        X_test = X_test / 255.0

    # Reshape for CNN (adding a channel dimension)
    X_train = np.expand_dims(X_train, axis=-1)
    if X_test is not None:
        X_test = np.expand_dims(X_test, axis=-1)

    return X_train, X_test