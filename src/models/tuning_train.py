import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from src.models.evaluate import evaluate_model
import os
from tqdm import tqdm


def tuning_train_model(model, dataset, device, val_ratio=0.1, num_epochs=5,
                       batch_size=128, learning_rate=0.005, optimizer_name='Adam',
                       save_dir='~/Models/custom_note_transcription'):

    save_dir = os.path.expanduser(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    validation_size = int(val_ratio * len(dataset))
    train_size = len(dataset) - validation_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    best_val_f1 = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_loader_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        for i, (inputs, labels) in enumerate(train_loader_progress):
            inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1], inputs.shape[2]).to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

            train_loader_progress.set_description(
                f"Epoch {epoch+1}/{num_epochs} - Training, Loss: {running_loss/ (i + 1):.3f}")

        model.eval()
        accuracy, recall, precision, avg_val_f1 = evaluate_model(model, validation_loader, device, return_metrics=True)
        print(f'Epoch [{epoch + 1}/{num_epochs}] - Validation F1: {avg_val_f1:.3f}')

        # Save model if it improves on the best found F1 score
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            best_model_path = os.path.join(save_dir, 'best_model_state.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved best model state to {best_model_path}')

    return avg_val_f1
