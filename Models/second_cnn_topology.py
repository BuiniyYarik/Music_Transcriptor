from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, ReLU


def create_second_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(8, kernel_size=(3, 3), input_shape=input_shape, padding='same'),  # Added padding
        BatchNormalization(),
        ReLU(),
        Conv2D(8, (3, 3), padding='same'),  # Added padding
        BatchNormalization(),
        ReLU(),
        MaxPooling2D(pool_size=(1, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='sigmoid')
    ])
    return model
