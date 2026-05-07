import h5py
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'Deploy', 'models', 'inceptionNet_model.h5')

try:
    f = h5py.File(MODEL_PATH, 'r')
    print("Keys:", list(f.keys()))
    if 'model_weights' in f:
        mw = f['model_weights']
        print("Model Weights Keys:", list(mw.keys()))
        for key in mw.keys():
            print(f"  Layer: {key}, Weights: {list(mw[key].keys())}")
        
    if 'keras_version' in f.attrs:
        print("Keras Version:", f.attrs['keras_version'])
        
except Exception as e:
    print(f"Error inspecting model: {e}")
