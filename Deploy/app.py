from flask import Flask, render_template, request, flash, redirect, url_for, send_from_directory, jsonify
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Masking
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import imageio
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from werkzeug.utils import secure_filename
from uuid import uuid4

try:
    from mtcnn import MTCNN
except Exception as e:
    print(f"MTCNN import failed, falling back to OpenCV Haar face detection: {e}")
    MTCNN = None

app = Flask(__name__)
app.secret_key = "secret key"

# Configure upload folder
# Configure upload folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
REMOTE_DOWNLOAD_LIMIT = app.config['MAX_CONTENT_LENGTH']


@app.after_request
def add_extension_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response


def is_allowed_video(filename):
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_VIDEO_EXTENSIONS


def unique_upload_name(filename):
    clean_name = secure_filename(filename) or 'video.mp4'
    return f"{uuid4().hex}_{clean_name}"


def download_video(url):
    parsed = urlparse(url)
    if parsed.scheme not in {'http', 'https'}:
        raise ValueError('Only http and https video URLs are supported.')

    basename = os.path.basename(parsed.path) or 'remote-video.mp4'
    if not is_allowed_video(basename):
        basename = f"{basename}.mp4"

    filename = unique_upload_name(basename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    request_headers = {'User-Agent': 'DeepFake-Detection-Extension/1.0'}

    with urlopen(Request(url, headers=request_headers), timeout=30) as response:
        total = 0
        with open(file_path, 'wb') as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > REMOTE_DOWNLOAD_LIMIT:
                    output.close()
                    os.remove(file_path)
                    raise ValueError('Remote video is larger than the configured limit.')
                output.write(chunk)

    return filename, file_path

# -------------------------------------------------------------------
# Model Construction
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'inceptionNet_model.h5')
FRAME_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'efficientnetv2_frame_model.keras')
MODEL_BACKBONE = 'InceptionV3'
MODEL_INPUT_SIZE = (299, 299)
FRAME_MODEL_INPUT_SIZE = (224, 224)
FRAME_MODEL_SEQUENCE_LENGTH = 32
ACTIVE_MODEL_NAME = 'InceptionV3 + GRU'
MIN_FACE_FRAMES = 8
MIN_FACE_FRAME_RATIO = 0.40

frame_model = None
if os.path.exists(FRAME_MODEL_PATH):
    try:
        print(f"Loading stronger frame model from {FRAME_MODEL_PATH}...")
        frame_model = tf.keras.models.load_model(FRAME_MODEL_PATH, compile=False)
        ACTIVE_MODEL_NAME = 'EfficientNetV2 frame ensemble'
        print("Frame model loaded.")
    except Exception as e:
        print(f"Error loading frame model, falling back to InceptionV3 + GRU: {e}")
        frame_model = None

def build_classifier_model():
    """
    Reconstructs the model architecture based on inspection.
    Architecture:
    Input(20, 2048) -> GRU(16) -> GRU(8) -> Dropout(0.4) -> Dense(8) -> Dense(2)
    Mask Input(20) is used for the first GRU.
    """
    try:
        input_features = Input(shape=(20, 2048), name='input_3')
        input_mask = Input(shape=(20,), dtype='bool', name='input_4')

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

# 1. Load Feature Extractor (InceptionV3 fallback)
feature_extractor = None
if frame_model is None:
    try:
        print("Loading InceptionV3 feature extractor...")
        feature_extractor = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
        print("InceptionV3 loaded.")
    except Exception as e:
        print(f"Error loading InceptionV3: {e}")
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
if frame_model is None:
    try:
        print("Building classifier model...")
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


mtcnn_detector = None
haar_face_cascade = None


def get_mtcnn_detector():
    global mtcnn_detector
    if MTCNN is None:
        return None
    if mtcnn_detector is None:
        print("Loading MTCNN face detector...")
        mtcnn_detector = MTCNN()
        print("MTCNN face detector loaded.")
    return mtcnn_detector


def get_haar_face_cascade():
    global haar_face_cascade
    if haar_face_cascade is None:
        haar_face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    return haar_face_cascade


def clamp_face_box(x, y, w, h, frame_shape):
    height, width = frame_shape[:2]
    x = max(0, int(x))
    y = max(0, int(y))
    w = max(0, int(w))
    h = max(0, int(h))
    w = min(width - x, w)
    h = min(height - y, h)
    return x, y, w, h


def detect_faces(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detector = get_mtcnn_detector()

    if detector is not None:
        try:
            detections = detector.detect_faces(rgb_frame)
            mtcnn_faces = []
            for detection in detections:
                confidence = float(detection.get('confidence', 0.0))
                if confidence < 0.80:
                    continue
                x, y, w, h = detection.get('box', [0, 0, 0, 0])
                x, y, w, h = clamp_face_box(x, y, w, h, frame.shape)
                if w > 0 and h > 0:
                    mtcnn_faces.append({
                        'box': (x, y, w, h),
                        'confidence': confidence,
                        'detector': 'mtcnn',
                    })
            if mtcnn_faces:
                return mtcnn_faces, 'mtcnn'
        except Exception as e:
            print(f"MTCNN detection failed, using Haar fallback: {e}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = get_haar_face_cascade().detectMultiScale(gray, 1.1, 4)
    haar_faces = []
    for x, y, w, h in faces:
        x, y, w, h = clamp_face_box(x, y, w, h, frame.shape)
        if w > 0 and h > 0:
            haar_faces.append({
                'box': (x, y, w, h),
                'confidence': None,
                'detector': 'haar',
            })
    return haar_faces, 'haar' if haar_faces else 'none'


def normalize_face_box(box, frame_shape):
    x, y, w, h = box
    height, width = frame_shape[:2]
    if width <= 0 or height <= 0:
        return None
    return {
        'cx': (x + (w / 2)) / width,
        'cy': (y + (h / 2)) / height,
        'area': (w * h) / (width * height),
    }


def build_temporal_analysis(face_tracks, visual_changes, sequence_length, frames_with_faces):
    valid_tracks = [track for track in face_tracks if track is not None]
    warnings = []

    if len(valid_tracks) < 3:
        return {
            'enabled': True,
            'method': 'face_track_consistency',
            'risk_level': 'not_available',
            'summary': 'Temporal analysis needs at least 3 sampled frames with a detectable face.',
            'warnings': warnings,
        }

    center_motion = []
    size_change = []
    previous = None
    for current in face_tracks:
        if current is None:
            previous = None
            continue
        if previous is not None:
            dx = current['cx'] - previous['cx']
            dy = current['cy'] - previous['cy']
            center_motion.append(float(np.sqrt((dx * dx) + (dy * dy))))
            if previous['area'] > 0:
                size_change.append(float(abs(current['area'] - previous['area']) / previous['area']))
        previous = current

    mean_motion = float(np.mean(center_motion)) if center_motion else 0.0
    max_motion = float(np.max(center_motion)) if center_motion else 0.0
    mean_size_change = float(np.mean(size_change)) if size_change else 0.0
    mean_visual_change = float(np.mean(visual_changes)) if visual_changes else 0.0
    visual_change_std = float(np.std(visual_changes)) if visual_changes else 0.0
    missing_face_ratio = float((sequence_length - frames_with_faces) / sequence_length)

    if missing_face_ratio > 0.35:
        warnings.append('The face disappears in several sampled frames.')
    if mean_motion > 0.16 or max_motion > 0.34:
        warnings.append('Face position changes sharply between sampled frames.')
    if mean_size_change > 0.45:
        warnings.append('Face size changes unusually across the video.')
    if visual_change_std > 0.20:
        warnings.append('Frame-to-frame visual changes are inconsistent.')

    if len(warnings) >= 2:
        risk_level = 'high'
    elif warnings:
        risk_level = 'medium'
    else:
        risk_level = 'low'

    summary_by_risk = {
        'low': 'Face movement is mostly stable across sampled frames.',
        'medium': 'Some temporal inconsistency was detected; manual review is recommended.',
        'high': 'Multiple temporal inconsistencies were detected; treat this video as suspicious.',
    }

    return {
        'enabled': True,
        'method': 'face_track_consistency',
        'risk_level': risk_level,
        'summary': summary_by_risk[risk_level],
        'warnings': warnings,
        'metrics': {
            'mean_face_motion': mean_motion,
            'max_face_motion': max_motion,
            'mean_face_size_change': mean_size_change,
            'mean_visual_change': mean_visual_change,
            'visual_change_std': visual_change_std,
            'missing_face_ratio': missing_face_ratio,
        },
    }


def extract_frames(video_path, sequence_length=20, return_metadata=False):
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
             
        # Calculate indices
        skip = max(int(num_frames / sequence_length), 1)
        indices = [i * skip for i in range(sequence_length)]
        # Ensure we don't go out of bounds
        indices = [min(i, num_frames - 1) for i in indices]
        
        frames = []
        frames_with_faces = 0
        total_faces = 0
        detector_counts = {'mtcnn': 0, 'haar': 0, 'none': 0}
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                print(f"Error reading frame {i}")
                frames.append(np.zeros((MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0], 3), dtype='float32'))
                continue
                
            faces, detector_used = detect_faces(frame)
            detector_counts[detector_used] = detector_counts.get(detector_used, 0) + 1
            
            if len(faces) > 0:
                frames_with_faces += 1
                total_faces += len(faces)
                # Get the largest face
                face = max(faces, key=lambda item: item['box'][2] * item['box'][3])
                x, y, w, h = face['box']
                # Add some margin if possible
                margin = int(0.1 * w)
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(frame.shape[1] - x, w + 2 * margin)
                h = min(frame.shape[0] - y, h + 2 * margin)
                
                # Crop face
                frame = frame[y:y+h, x:x+w]
            
            # Convert to RGB (InceptionV3 expects RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to InceptionV3 input
            frame = cv2.resize(frame, MODEL_INPUT_SIZE)
            
            # Convert to float
            frame = frame.astype('float32')
            frames.append(frame)
            
        cap.release()
            
        # Pad if less than 20 frames
        while len(frames) < sequence_length:
            frames.append(np.zeros((MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0], 3), dtype='float32'))
            
        frames = np.array(frames)
        # Preprocess for InceptionV3 (scales to -1..1)
        frames = preprocess_input(frames)
        if return_metadata:
            metadata = {
                'sampled_frames': sequence_length,
                'frames_with_faces': frames_with_faces,
                'total_faces': total_faces,
                'face_frame_ratio': frames_with_faces / sequence_length,
                'face_detector': 'mtcnn' if detector_counts.get('mtcnn', 0) else 'haar',
                'detector_counts': detector_counts,
            }
            return frames, metadata
        return frames
        
    except Exception as e:
        print(f"Error extracting frames: {e}")
        traceback.print_exc()
        return None


def extract_frame_model_frames(video_path, sequence_length=FRAME_MODEL_SEQUENCE_LENGTH, return_metadata=False):
    """
    Extract raw RGB face crops for the EfficientNetV2 frame model.
    The trained model includes its own EfficientNetV2 preprocessing layer.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if num_frames == 0:
            print("Error: Video has 0 frames")
            cap.release()
            return None

        indices = np.linspace(0, num_frames - 1, sequence_length, dtype=np.int32)
        frames = []
        frames_with_faces = 0
        total_faces = 0
        detector_counts = {'mtcnn': 0, 'haar': 0, 'none': 0}

        for frame_index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ret, frame = cap.read()
            if not ret:
                frames.append(np.zeros((FRAME_MODEL_INPUT_SIZE[1], FRAME_MODEL_INPUT_SIZE[0], 3), dtype='float32'))
                continue

            faces, detector_used = detect_faces(frame)
            detector_counts[detector_used] = detector_counts.get(detector_used, 0) + 1

            if len(faces) > 0:
                frames_with_faces += 1
                total_faces += len(faces)
                face = max(faces, key=lambda item: item['box'][2] * item['box'][3])
                x, y, w, h = face['box']
                margin = int(0.18 * max(w, h))
                x, y, w, h = clamp_face_box(x - margin, y - margin, w + (2 * margin), h + (2 * margin), frame.shape)
                if w > 0 and h > 0:
                    frame = frame[y:y+h, x:x+w]

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, FRAME_MODEL_INPUT_SIZE)
            frames.append(frame.astype('float32'))

        cap.release()

        while len(frames) < sequence_length:
            frames.append(np.zeros((FRAME_MODEL_INPUT_SIZE[1], FRAME_MODEL_INPUT_SIZE[0], 3), dtype='float32'))

        frames = np.array(frames, dtype='float32')
        if return_metadata:
            metadata = {
                'sampled_frames': sequence_length,
                'frames_with_faces': frames_with_faces,
                'total_faces': total_faces,
                'face_frame_ratio': frames_with_faces / sequence_length,
                'face_detector': 'mtcnn' if detector_counts.get('mtcnn', 0) else 'haar',
                'detector_counts': detector_counts,
            }
            return frames, metadata
        return frames
    except Exception as e:
        print(f"Error extracting frame-model frames: {e}")
        traceback.print_exc()
        return None


def build_unsupported_content_result(content_metadata):
    frames_with_faces = int(content_metadata.get('frames_with_faces', 0))
    sampled_frames = int(content_metadata.get('sampled_frames', 0)) or 1
    face_frame_ratio = float(content_metadata.get('face_frame_ratio', frames_with_faces / sampled_frames))

    if frames_with_faces >= MIN_FACE_FRAMES and face_frame_ratio >= MIN_FACE_FRAME_RATIO:
        return None

    return {
        'status': 'unsupported',
        'label': 'NO HUMAN FACE DETECTED',
        'probability': None,
        'message': (
            'No clear human face was detected in enough frames. '
            'This can happen with animated, cartoon, gameplay, scenery, object-only, or very dark videos. '
            'Please upload a video with a visible human face for REAL/FAKE analysis.'
        ),
        'content': content_metadata,
        'requirements': {
            'minimum_face_frames': MIN_FACE_FRAMES,
            'minimum_face_frame_ratio': MIN_FACE_FRAME_RATIO,
        },
    }


def predict_with_frame_model(video_path):
    frame_result = extract_frame_model_frames(video_path, return_metadata=True)
    if frame_result is None:
        raise ValueError("Could not extract frames")

    frames, content_metadata = frame_result
    unsupported_result = build_unsupported_content_result(content_metadata)
    if unsupported_result is not None:
        return unsupported_result

    frame_probabilities = frame_model.predict(frames, verbose=0)
    prob_real = float(np.mean(frame_probabilities[:, 0]))
    prob_fake = float(np.mean(frame_probabilities[:, 1]))

    # Threshold is tunable for calibration.
    fake_threshold = float(os.getenv('DEEPFAKE_FAKE_THRESHOLD_FRAME', '0.5'))

    if prob_fake >= fake_threshold:
        label = 'FAKE'
        probability = prob_fake
    else:
        label = 'REAL'
        probability = prob_real

    return {
        'status': 'ok',
        'model': {
            'name': ACTIVE_MODEL_NAME,
            'classifier_weights': os.path.basename(FRAME_MODEL_PATH),
            'sequence_length': FRAME_MODEL_SEQUENCE_LENGTH,
            'aggregation': 'mean_frame_probability',
        },
        'label': label,
        'probability': probability,
        'probabilities': {
            'real': prob_real,
            'fake': prob_fake,
        },
        'threshold': fake_threshold,
        'content': content_metadata,
        'temporal_analysis': None,
    }


def predict_video_path(video_path):
    if frame_model is not None:
        return predict_with_frame_model(video_path)

    if feature_extractor is None or classifier_model is None:
        raise RuntimeError("Models not loaded properly.")

    frame_result = extract_frames(video_path, sequence_length=20, return_metadata=True)
    if frame_result is None:
        raise ValueError("Could not extract frames")

    frames, content_metadata = frame_result
    if frames is None:
        raise ValueError("Could not extract frames")

    unsupported_result = build_unsupported_content_result(content_metadata)
    if unsupported_result is not None:
        return unsupported_result

    features = feature_extractor.predict(frames)
    features_batch = np.expand_dims(features, axis=0)
    mask_batch = np.ones((1, 20), dtype=bool)

    prediction_prob = classifier_model.predict([features_batch, mask_batch])
    prob_real = float(prediction_prob[0][0])
    prob_fake = float(prediction_prob[0][1])
    fake_threshold = 0.4

    if prob_fake >= fake_threshold:
        label = 'FAKE'
        probability = prob_fake
    else:
        label = 'REAL'
        probability = prob_real

    return {
        'status': 'ok',
        'model': {
            'backbone': MODEL_BACKBONE,
            'classifier_weights': os.path.basename(MODEL_PATH),
            'sequence_length': 20,
            'feature_size': 2048,
        },
        'label': label,
        'probability': probability,
        'probabilities': {
            'real': prob_real,
            'fake': prob_fake,
        },
        'threshold': fake_threshold,
        'content': content_metadata,
    }

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
    
    if file and is_allowed_video(file.filename):
        filename = unique_upload_name(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        flash('Video successfully uploaded and displayed below')
        return render_template('upload.html', filename=filename)
    
    flash('Allowed video types are mp4, avi, mov')
    return redirect(request.url)

@app.route('/display/<filename>')
def display_video(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/health')
def api_health():
    models_loaded = frame_model is not None or (feature_extractor is not None and classifier_model is not None)
    return jsonify({
        'status': 'ok',
        'models_loaded': models_loaded,
        'model': {
            'active': ACTIVE_MODEL_NAME,
            'backbone': 'EfficientNetV2B0' if frame_model is not None else MODEL_BACKBONE,
            'classifier_weights': os.path.basename(FRAME_MODEL_PATH if frame_model is not None else MODEL_PATH),
            'sequence_length': FRAME_MODEL_SEQUENCE_LENGTH if frame_model is not None else 20,
            'feature_size': None if frame_model is not None else 2048,
        },
    })

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def api_predict():
    if request.method == 'OPTIONS':
        return ('', 204)

    try:
        filename = None
        file_path = None

        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'status': 'error', 'error': 'No video file selected.'}), 400
            if not is_allowed_video(file.filename):
                return jsonify({'status': 'error', 'error': 'Unsupported video type.'}), 400

            filename = unique_upload_name(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
        else:
            payload = request.get_json(silent=True) or {}
            url = payload.get('url')
            if not url:
                return jsonify({'status': 'error', 'error': 'Send a video file or a JSON body with a url.'}), 400
            filename, file_path = download_video(url)

        result = predict_video_path(file_path)
        result.update({
            'filename': filename,
        })
        return jsonify(result)
    except Exception as e:
        print(f"API prediction error: {e}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/sequence_prediction/<filename>')
def sequence_prediction(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        result = predict_video_path(video_path)
        if result.get('status') == 'unsupported':
            return render_template(
                'upload.html',
                filename=filename,
                prediction=result['message'],
                prediction_status='unsupported'
            )
        return render_template(
            'upload.html',
            filename=filename,
            prediction=f"{result['label']} (Probability: {result['probability']:.2f})",
            prediction_status=result['label'].lower()
        )

    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('upload.html', filename=filename, prediction=f"Error: {e}", prediction_status='error')

if __name__ == "__main__":
    app.run(debug=True)
