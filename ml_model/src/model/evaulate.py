from ml_model.src.data.dataloader import load_data
from utils.preprocess import preprocess_data

def evaluate_model(model, test_dir, label_encoder):
    # Load test data
    _, _, X_test, y_test, _ = load_data('', test_dir)

    # Preprocess data
    X_test = preprocess_data(X_test)

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")