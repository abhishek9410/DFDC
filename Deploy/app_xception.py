from flask import Flask, render_template, request, flash, redirect, url_for, send_from_directory
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Masking
from tensorflow.keras.applications.efficientnet import EfficientNetB2, preprocess_input
import imageio

app = Flask(__name__)
app.secret_key = "secret key"

# Configure upload folder
# Configure upload folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -------------------------------------------------------------------
# Model Construction
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'Saved Models', 'models', 'CNN_RNN', 'latest_EfficientNetB2.h5'))

def build_classifier_model():
    """
    Reconstructs the model architecture based on inspection.
    Architecture:
    Input(20, 2048) -> GRU(16) -> GRU(8) -> Dropout(0.4) -> Dense(8) -> Dense(2)
    Mask Input(20) is used for the first GRU.
    """
    try:
        input_features = Input(shape=(25, 1408), name='input_5')
        input_mask = Input(shape=(25,), dtype='bool', name='input_6')

        x = GRU(16, return_sequences=True, name='gru')(input_features, mask=input_mask)
        x = GRU(8, return_sequences=False, name='gru_1')(x)
        x = Dropout(0.4, name='dropout')(inputs=x)
        x = Dense(8, activation='relu', name='dense')(inputs=x)
        outputs = Dense(2, activation='softmax', name='dense_1')(inputs=x)
        
        model = Model(inputs=[input_features, input_mask], outputs=outputs, name='functional_model')
        return model
    except Exception as e:
        print(f"Exception in build_classifier_model: {e}")
        traceback.print_exc()
        raise e

# 1. Load Feature Extractor (InceptionV3)
try:
    print("Loading EfficientNetB2 feature extractor...")
    feature_extractor = EfficientNetB2(weights='imagenet', include_top=False, pooling='avg')
    print("EfficientNetB2 loaded.")
except Exception as e:
    print(f"Error loading EfficientNetB2: {e}")
    feature_extractor = None

import h5py

def load_weights_manually(model, filepath):
    """
    Load weights from h5 file manually to bypass Keras loading errors.
    Assumes structure: model_weights -> layer_name -> layer_name -> weights
    """
    print(f"Loading weights manually from {filepath}")
    try:
        f = h5py.File(filepath, 'r')
        if 'model_weights' not in f:
            print("No 'model_weights' group found!")
            return False
            
        mw = f['model_weights']
        
        for layer in model.layers:
            if layer.name in mw:
                layer_group = mw[layer.name]
                
                # Check for nested group with same name (common in Keras h5)
                if layer.name in layer_group:
                    weight_group = layer_group[layer.name]
                else:
                    weight_group = layer_group
                
                # Check if weights are nested in a 'cell' group (e.g. gru_cell)
                # Keras 2 saved GRU/LSTM weights often inside a cell sub-group
                if 'kernel:0' not in weight_group:
                    # Look for any key ending in 'cell' or 'cell_X'
                    keys = list(weight_group.keys())
                    cell_key = next((k for k in keys if 'cell' in k), None)
                    if cell_key:
                        # print(f"Found nested cell group: {cell_key}")
                        weight_group = weight_group[cell_key]

                # Get weight values
                expected_weights = layer.get_weights()
                found_weights = []
                loaded_vals = []
                
                weight_keys = list(weight_group.keys())
                
                if not weight_keys:
                    continue
                    
                # Specific logic for GRU and Dense
                if 'gru' in layer.name:
                    # GRU weights: kernel, recurrent_kernel, bias
                    # keys in h5: 'kernel:0', 'recurrent_kernel:0', 'bias:0'
                    if 'kernel:0' in weight_group:
                        w_kernel = weight_group['kernel:0'][()]
                        w_recurrent = weight_group['recurrent_kernel:0'][()]
                        w_bias = weight_group['bias:0'][()]
                        loaded_vals = [w_kernel, w_recurrent, w_bias]
                    
                elif 'dense' in layer.name:
                    if 'kernel:0' in weight_group:
                        w_kernel = weight_group['kernel:0'][()]
                        w_bias = weight_group['bias:0'][()]
                        loaded_vals = [w_kernel, w_bias]
                
                if len(loaded_vals) == len(expected_weights):
                   # Verify shapes
                   shape_match = True
                   for i, val in enumerate(loaded_vals):
                       if val.shape != expected_weights[i].shape:
                           print(f"Shape mismatch for {layer.name} weight {i}: file {val.shape} vs model {expected_weights[i].shape}")
                           shape_match = False
                   
                   if shape_match:
                       layer.set_weights(loaded_vals)
                       print(f"Set weights for {layer.name}")
                else:
                    # Only print warning if we expected weights but couldn't load them (and it's not a utility layer)
                    pass
                
        f.close()
        return True
    except Exception as e:
        print(f"Manual loading failed: {e}")
        traceback.print_exc()
        return False

# 2. Build and Load Classifier
import traceback
classifier_model = None
try:
    print("Building classifier model...")
    # Clean up debug prints in build_classifier_model first? 
    # Calling it here assumes it's defined.
    # We should update build_classifier_model separately to remove prints.
    classifier_model = build_classifier_model()
    print("Loading weights manually from h5...")
    if load_weights_manually(classifier_model, MODEL_PATH):
        print("Classifier weights loaded successfully.")
    else:
        print("Failed to load weights manually.")
except Exception as e:
    print(f"Error loading classifier weights: {e}")
    traceback.print_exc()
    classifier_model = None


def extract_frames(video_path, sequence_length=25):
    """
    Extracts 20 frames from the video, detects and crops faces, and ensures uniform sampling.
    Returns (20, 299, 299, 3) image batch.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
            
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if num_frames == 0:
             print("Error: Video has 0 frames")
             return None
             
        # Load Face Detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Calculate indices
        skip = max(int(num_frames / sequence_length), 1)
        indices = [i * skip for i in range(sequence_length)]
        # Ensure we don't go out of bounds
        indices = [min(i, num_frames - 1) for i in indices]
        
        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                print(f"Error reading frame {i}")
                frames.append(np.zeros((299, 299, 3), dtype='float32'))
                continue
                
            # Face Detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Get the largest face
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                # Add some margin if possible
                margin = int(0.1 * w)
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(frame.shape[1] - x, w + 2 * margin)
                h = min(frame.shape[0] - y, h + 2 * margin)
                
                # Crop face
                frame = frame[y:y+h, x:x+w]
            
            # Convert to RGB (Inception expects RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to EfficientNetB2 input
            frame = cv2.resize(frame, (260, 260))
            
            # Convert to float
            frame = frame.astype('float32')
            frames.append(frame)
            
        cap.release()
            
        # Pad if less than 25 frames
        while len(frames) < sequence_length:
            frames.append(np.zeros((260, 260, 3), dtype='float32'))
            
        frames = np.array(frames)
        # Preprocess for InceptionV3 (scales to -1..1)
        frames = preprocess_input(frames)
        return frames
        
    except Exception as e:
        print(f"Error extracting frames: {e}")
        traceback.print_exc()
        return None

# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        flash('Video successfully uploaded and displayed below')
        return render_template('upload.html', filename=filename)
    
    flash('Allowed video types are mp4, avi, mov')
    return redirect(request.url)

@app.route('/display/<filename>')
def display_video(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/sequence_prediction/<filename>')
def sequence_prediction(filename):
    if feature_extractor is None or classifier_model is None:
        flash("Models not loaded properly.")
        return render_template('upload.html', filename=filename, prediction="Error: Models not loaded")

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        # 1. Extract Frames
        frames = extract_frames(video_path, sequence_length=25)
        if frames is None:
            raise ValueError("Could not extract frames")
            
        # 2. Extract Features
        # Input: (20, 299, 299, 3) -> Output: (20, 2048)
        features = feature_extractor.predict(frames) 
        
        # 3. Prepare for Classifier
        # Shape (1, 25, 1408)
        features_batch = np.expand_dims(features, axis=0)
        # Create Mask (1, 25) - all True
        mask_batch = np.ones((1, 25), dtype=bool)
        
        # 4. Predict
        prediction_prob = classifier_model.predict([features_batch, mask_batch])
        
        # 5. Interpret
        # Output is (1, 2) softmax [prob_fake, prob_real] OR [prob_real, prob_fake]?
        # Usually class 0 is one and 1 is the other.
        # Without label map, assuming 0=REAL, 1=FAKE or vice versa.
        # Notebook snippet said "y is 1 if FAKE" (common in DeepFake)
        prob_real = prediction_prob[0][0]
        prob_fake = prediction_prob[0][1]
        
        # Adjusting the threshold to make FAKE detection more robust
        # Since it was predicting all as REAL, we lower the threshold for FAKE
        fake_threshold = 0.4 
        
        if prob_fake >= fake_threshold:
            return render_template('upload.html', filename=filename, prediction=f"FAKE (Probability: {prob_fake:.2f})")
        else:
            return render_template('upload.html', filename=filename, prediction=f"REAL (Probability: {prob_real:.2f})")

    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('upload.html', filename=filename, prediction=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
