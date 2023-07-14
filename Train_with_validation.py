 import numpy as np
import cv2
from tensorflow import keras
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.applications.vgg19 import VGG19
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold

dataset_dir = 'augmented_images'
dataset = Path(dataset_dir)

# Number images dictionary
image_dict = {
    '1': list(dataset.glob('1/*.jpg')),
    '2': list(dataset.glob('2/*.jpg')),
    '3': list(dataset.glob('3/*.jpg'))
}

image_labels = {
    '1': 0,
    '2': 1,
    '3': 2
}

# Load the images and labels
X, y = [], []
for image_name, images in image_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        img_resize = cv2.resize(img, (224, 224))
        X.append(img_resize)
        y.append(image_labels[image_name])

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Convert labels to one-hot encoded vectors
y = keras.utils.to_categorical(y, 3)

# Scale the images
X_scale = X / 255

# Define the number of folds for K-fold cross-validation
n_splits = 5

# Create K-fold cross-validation iterator
kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

# Initialize variables to store validation accuracy for each fold
val_accs = []

# Loop over the folds
for i, (train_idx, val_idx) in enumerate(kf.split(X_scale)):
    print(f'Fold {i+1}/{n_splits}')

    # Split the data into training and validation sets
    X_train, y_train = X_scale[train_idx], y[train_idx]
    X_val, y_val = X_scale[val_idx], y[val_idx]

    # Load the VGG19 model
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the layers of the pre-trained model
    for layer in base_model.layers:
        layer.trainable = False

    # Add new trainable layers on top of the pre-trained model
    x = base_model.output
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    predictions = Dense(3, activation='softmax')(x)

    # Define the new model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model on the training set
    history = model.fit(
        X_train,
        y_train,
        epochs=5,
        verbose=1,
        validation_data=(X_val, y_val)
    )
    model.save('my_model.h5')
    # Evaluate the model on the validation set
    val_loss, val_acc = model.evaluate(X_val, y_val)

    # Store the validation accuracy for this fold
    val_accs.append(val_acc)

# Calculate the mean and standard deviation of the validation accuracies
mean_val_acc = np.mean(val_accs)
std_val_acc = np.std(val_accs)

# Print the results
print(f"Mean validation accuracy: {mean_val_acc:.4f}")
print(f"Standard deviation of validation accuracy: {std_val_acc:.4f}")
