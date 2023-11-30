# Importing libraries needed for Operating System Manipulation in Python
import os

# Importing libraries  for file names and paths
from pathlib import Path

# Importing library for manipulation time
from time import strftime

# Import libraries TensorFlow e Keras
import tensorflow as tf

# Setting log path
root_logdir = os.path.join(os.curdir, "logs")

def get_run_logdir(root_logdir="logs"):
    return Path(root_logdir) / strftime("run_%Y_%m_%d_%H_%M_%S")

def name_logdir(name):
    # Increment every time you train the model
    run_index = 1
    run_logdir = Path() / "logs" / f"{name}"

    # Return Callback TensorBoard()
    return tf.keras.callbacks.TensorBoard(run_logdir)

# Function call
#run_logdir = get_run_logdir()
