from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense


def create_first_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='same'),  # Adjusted kernel size and added padding
        Conv2D(64, (3, 3), activation='relu', padding='same'),  # Adjusted kernel size and added padding
        MaxPooling2D(pool_size=(1, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model
