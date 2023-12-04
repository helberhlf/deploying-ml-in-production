#-------------------------------------------------------
# @author Helber
#-------------------------------------------------------

# Importing libraries needed for Operating System Manipulation in Python
import os

# Import DL livraries (APIs) bulding up DL pipelines and AutoDL livraries (APIs) for tuning DL pipelines
import tensorflow as tf
import keras_tuner as kt
#-------------------------------------------------------

""""A named constant is a name that represents a value that cannot be 
changed during the program's execution."""

# Using Global Constants Defining Named Constants
INPUT = 12

# Creating a tuner called DeepTuner by extending the base tuner class
class DeepTuner(kt.Tuner):
    def run_trial(self, trial, X, y, validation_data, **fit_kwargs):
        model = self.hypermodel.build(trial.hyperparameters)
        model.fit(X, y,
                  batch_size=trial.hyperparameters.Choice("batch_size", [32,64]),
                  **fit_kwargs  # Trains model with a tunable batch size
                  )
        # get the validation data
        X_val, y_val = validation_data
        eval_scores = model.evaluate(X_val, y_val)

        # save the model to disk
        self.save_model(trial.trial_id, model)

        # inform the oracle of the eval result, the result is a dictionary with the metric names as the keys.
        return {
            name: value for name, value in zip(model.metrics_names, eval_scores)
        }
    '''
    Since TensorFlow Keras provides methods to save and load the models, 
    we can adopt these methods to implement the save_model() and load_model() functions.
    '''
    def save_model(self, trial_id, model, step=0):
        fname = os.path.join(self.get_trial_dir(trial_id), "../model")
        model.save(fname)
    def load_model(self, trial):
        fname = os.path.join(self.get_trial_dir(trial.trial_id), "../model")
        model = tf.keras.models.load_model(fname)

        return model
#-------------------------------------------------------

# Creates a search space for tuning MLPs
# Define the search space for classifier
def build_classifier(hp):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(INPUT,)))

    # Tune the number of layers.
    for i in range(hp.Int("num_layers", min_value=1, max_value=4)):
        model.add(
            tf.keras.layers.Dense(
                # Tune number of units separately.
                units=hp.Int(f"units_{i}", min_value=32, max_value=256, step=64),
                activation=hp.Choice("activation", ["elu","gelu","relu","swish"]),
                kernel_initializer=hp.Choice("kernel_initializer", ["he_normal"]),
            )
        )
    # Tune whether to use dropout
    #if hp.Boolean("dropout"):
    #    model.add(tf.keras.layers.Dropout(hp.Float("dropout_rate",min_value=0.2, max_value=0.4, step=0.1)))
    model.add(tf.keras.layers.Dropout(hp.Float("dropout_rate", min_value=0.2, max_value=0.4, step=0.1)))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid",))

    # Instantiates the hp.Choice() method to select the optimization method and learning rate
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
    optimizer_name = hp.Choice("optimizer", ["adam", "nadam"])
    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)

    # Compile the model
    model.compile(optimizer=optimizer,  # Defines the search space for the optimizer
                  loss='binary_crossentropy',  # Compiles the model set the loss
                  metrics=['accuracy', 'Precision', 'Recall', 'AUC'])  # the metric we're using is ACC, PRE ,REC, AUC.

    # Return the model is keras
    return model
