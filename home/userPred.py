import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import librosa
from keras.preprocessing.sequence import pad_sequences


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
    

# Load saved models
loaded_cnn_model = tf.keras.models.load_model("home/cnn_model.h5")
loaded_dcnn_model = tf.keras.models.load_model("home/dcnn_model.h5")

# Load label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('home/label_encoder_classes.npy')


def new_prediction(file):

    # Example: Predict emotion for a new audio file
    new_audio_file_path = file  # Replace with the path to your new audio file
    new_features = extract_features(new_audio_file_path)

    result = None

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

        result = predicted_label[0]

    
    return result






# # Example: Predict emotion for a new audio file
# new_audio_file_path = 'home/WhatsApp Ptt 2024-02-17 at 8.57.55 AM.ogg'  # Replace with the path to your new audio file
# new_features = extract_features(new_audio_file_path)

# if new_features is not None:
#     # Reshape the features to match the input shape of the models
#     new_features = new_features.reshape(1, *new_features.shape, 1)

#     # Make predictions using the loaded models
#     cnn_prediction = loaded_cnn_model.predict(new_features)
#     dcnn_prediction = loaded_dcnn_model.predict(new_features)

#     # Combine predictions (e.g., averaging)
#     ensemble_prediction = 0.5 * cnn_prediction + 0.5 * dcnn_prediction

#     # Convert predictions to emotion labels using label encoder
#     predicted_label_idx = np.argmax(ensemble_prediction, axis=1)
#     predicted_label = label_encoder.inverse_transform(predicted_label_idx)

    print(f"The predicted emotion for the new audio file is: {predicted_label[0]}")





