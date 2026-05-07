import tensorflow as tf
import sys
print(f"Python Version: {sys.version}")
print(f"TensorFlow Version: {tf.__version__}")
try:
    import keras
    print(f"Keras Version: {keras.__version__}")
except:
    print("Keras not found directly")

print(f"tf.keras.version: {getattr(tf.keras, '__version__', 'N/A')}")
