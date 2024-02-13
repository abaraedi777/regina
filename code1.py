import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras import layers, models
from keras.utils import to_categorical
import librosa
from keras.preprocessing.sequence import pad_sequences

print("started")

# Function to extract features from audio files
def extract_features(file_path, max_sequence_length=40):
    try:
        audio_data, _ = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio_data, sr=22050, n_mfcc=13)

        # Pad or truncate features to have a consistent shape
        padded_features = pad_sequences([mfccs.T], maxlen=max_sequence_length, padding='post').T
        return padded_features

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

# Load audio dataset
audio_data = []
labels = []

# path to dataset
dataset_path = 'dataset1'
classes = os.listdir(dataset_path)

max_sequence_length = 40  # Set the desired sequence length

for emotion_class in classes:
    class_path = os.path.join(dataset_path, emotion_class)
    for file_name in os.listdir(class_path):
        file_path = os.path.join(class_path, file_name)
        features = extract_features(file_path, max_sequence_length)
        if features is not None:
            audio_data.append(features)
            labels.append(emotion_class)

# Convert features and labels to numpy arrays
X = np.array(audio_data)
y = np.array(labels)

# Convert labels to numeric format
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Calculate the number of classes
num_classes = len(set(encoded_labels))

# Convert labels to one-hot encoding
y = to_categorical(encoded_labels, num_classes)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CNN model
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define the DCNN model
def create_dcnn_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', dilation_rate=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build and train the models
input_shape = X_train.shape[1:]
label_encoder = LabelEncoder()
y_train_numeric = label_encoder.fit_transform(np.argmax(y_train, axis=1))
num_classes = len(label_encoder.classes_)

cnn_model = create_cnn_model(input_shape, num_classes)
dcnn_model = create_dcnn_model(input_shape, num_classes)

# Train models
cnn_model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
dcnn_model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Save models
cnn_model.save("cnn_model.h5")
dcnn_model.save("dcnn_model.h5")

# Save label encoder
np.save('label_encoder_classes.npy', label_encoder.classes_)

# Load saved models
loaded_cnn_model = tf.keras.models.load_model("cnn_model.h5")
loaded_dcnn_model = tf.keras.models.load_model("dcnn_model.h5")

# Load label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder_classes.npy')

# Example: Predict emotion for a new audio file
new_audio_file_path = '1AnQA052.wav'  # Replace with the path to your new audio file
new_features = extract_features(new_audio_file_path)

if new_features is not None:
    # Reshape the features to match the input shape of the models
    new_features = new_features.reshape(1, *new_features.shape, 1)

    # Make predictions using the loaded models
    cnn_prediction = loaded_cnn_model.predict(new_features)
    dcnn_prediction = loaded_dcnn_model.predict(new_features)

    # Combine predictions (e.g., averaging)
    ensemble_prediction = 0.5 * cnn_prediction + 0.5 * dcnn_prediction

    # Convert predictions to emotion labels using label encoder
    predicted_label = label_encoder.inverse_transform(np.argmax(ensemble_prediction, axis=1))

    print(f"The predicted emotion for the new audio file is: {predicted_label[0]}")

    # Assuming label_encoder is already loaded
    real_predicted_label = label_encoder.inverse_transform([predicted_label[0]])
    print(f"The predicted emotion for the new audio file is: {real_predicted_label[0]}")

