import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np


def evaluate_model(model, val_loader, device, threshold=0.5, return_metrics=False):
    """
    Evaluates the model on the validation set.

    Parameters:
    model (torch.nn.Module): Trained model
    val_loader (torch.utils.data.DataLoader): Validation set loader
    device (torch.device): Device on which to perform computation
    threshold (float): Threshold to use for predictions
    return_metrics (bool): Whether to return the metrics or not

    Returns:
    accuracy (float): Accuracy score
    recall (float): Recall score
    precision (float): Precision score
    f1 (float): F1 score
    """
    model.eval()
    val_predictions = []
    val_true_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1], inputs.shape[2]).to(device)
            labels = labels.to(device).float()

            outputs = model(inputs)
            predictions = outputs > threshold  # Apply threshold
            val_predictions.extend(predictions.cpu().numpy())
            val_true_labels.extend(labels.cpu().numpy())

    # Convert to binary format for evaluation
    val_predictions = np.array(val_predictions).astype(int)
    val_true_labels = np.array(val_true_labels).astype(int)

    # Calculate the metrics
    accuracy = accuracy_score(val_true_labels, val_predictions)
    recall = recall_score(val_true_labels, val_predictions, average='weighted', zero_division=1)
    precision = precision_score(val_true_labels, val_predictions, average='weighted', zero_division=1)
    f1 = f1_score(val_true_labels, val_predictions, average='weighted', zero_division=1)

    print(f'Test Accuracy: {accuracy}')
    print(f'Test Recall: {recall}')
    print(f'Test Precision: {precision}')
    print(f'Test F1: {f1}')

    if return_metrics:
        return accuracy, recall, precision, f1
