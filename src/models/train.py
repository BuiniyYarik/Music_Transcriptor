import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm


def train_model(model, dataset, device, val_ratio=0.1, num_epochs=5,
                batch_size=128, learning_rate=0.005,
                save_dir='~/Models/custom_note_transcription'):
    """
    Train the model

    Parameters:
    model (nn.Module): Model to train
    dataset (AudioDataset): Dataset to use for training
    device (torch.device): Device on which to perform computation
    val_ratio (float): Ratio of the dataset to use for validation
    num_epochs (int): Number of epochs to train the model for
    batch_size (int): Batch size
    learning_rate (float): Learning rate
    save_dir (str): Directory to save the model checkpoints to

    Returns:
    None
    """
    # Ensure save directory exists
    save_dir = os.path.expanduser(save_dir)  # Expands the user home directory symbol '~'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    validation_size = int(val_ratio * len(dataset))
    train_size = len(dataset) - validation_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    best_val_loss = float('inf')
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

            # Update progress bar description with the running loss
            train_loader_progress.set_description(
                f"Epoch {epoch+1}/{num_epochs} - Training, Loss: {running_loss/ (i + 1):.3f}")

        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        validation_loss = 0.0
        validation_loader_progress = tqdm(validation_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation")
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(validation_loader_progress):
                inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1], inputs.shape[2]).to(device)
                labels = labels.to(device).float()
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                validation_loss += loss.item()

                # Update progress bar description with the running loss
                validation_loader_progress.set_description(
                    f"Epoch {epoch+1}/{num_epochs} - Validation, Loss: {validation_loss/ (i + 1):.3f}")

        avg_val_loss = validation_loss / len(validation_loader)

        print(f'Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_train_loss:.3f}, Validation Loss: {avg_val_loss:.3f}')

        # Save checkpoint if the model improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(save_dir, 'best_model_state.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved best model state to {best_model_path}')

        # Save the latest model state
        latest_checkpoint_path = os.path.join(save_dir, 'latest_model_checkpoint.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_val_loss,
        }, latest_checkpoint_path)
        print(f'Saved latest model checkpoint to {latest_checkpoint_path}')

    print('Finished Training')
