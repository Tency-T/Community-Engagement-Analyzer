import os
import tensorflow as tf

# Set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Configure TensorFlow threading
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
