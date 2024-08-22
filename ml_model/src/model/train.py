from ml_model.src.data.dataloader import load_data
from ml_model.src.model import create_cnn_model
from ml_model.utils.preprocess import preprocess_data
from ml_model.utils.visualize import plot_training_history

import tensorflow as tf

def train_model(train_dir, test_dir, epochs, batch_size):
    # Load data
    X_train, y_train, X_test, y_test, label_encoder = load_data(train_dir, test_dir)

    # Preprocess data
    X_train, X_test = preprocess_data(X_train, X_test)

    # Create model
    model = create_cnn_model(X_train.shape[1:])

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    # Plot training history
    plot_training_history(history)

    return model, label_encoder