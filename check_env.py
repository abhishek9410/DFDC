import sys
print(f"Python: {sys.version}")

try:
    import h5py
    print(f"h5py: {h5py.__version__}")
except Exception as e:
    print(f"h5py import failed: {e}")

try:
    import numpy
    print(f"numpy: {numpy.__version__}")
except Exception as e:
    print(f"numpy import failed: {e}")

try:
    import cv2
    print(f"cv2: {cv2.__version__}")
except Exception as e:
    print(f"cv2 import failed: {e}")

try:
    import flask
    print(f"flask: {flask.__version__}")
except Exception as e:
    print(f"flask import failed: {e}")

print("Importing tensorflow...")
try:
    import tensorflow as tf
    print(f"TensorFlow: {tf.__version__}")
    print(f"Keras (from tf): {tf.keras.__version__}")
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    print("InceptionV3 imported successfully")
except Exception as e:
    print(f"TensorFlow/InceptionV3 import failed: {e}")
    import traceback
    traceback.print_exc()
