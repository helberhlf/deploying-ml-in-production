#-------------------------------------------------------
# @author Helber
#-------------------------------------------------------

# Importing libraries needed for Operating System Manipulation in Python
import os, sys

# Importing library for manipulation and exploration of datasets.
import numpy as np
import pandas as pd

from pathlib import Path
from time import strftime

# Importing libraries needed for predictive modeling, Machine Learning and Deep Learning algorithms..
import xgboost as xgb
from xgboost import plot_tree

# Import libraries TensorFlow e Keras
import tensorflow as tf
# Importing the progress bar
from tqdm.keras import TqdmCallback
#-------------------------------------------------------

# Machine Learning

# Creating a function to select the best features
def feature_imp(features, target,param_imp,n_best_features):
    # Define o classificador, ou seja, instância um objeto da classe XGBRegressor
    reg_XBGR = xgb.XGBRFRegressor(verbosity=0, silent=True)

    # ajuste os dados
    reg_XBGR.fit(features, target)

    # selecionando os melhores parâmetros com grid search, que indicar a importância relativa de cada atributo para fazer previsões precisas:
    reg_XBGR_feature_imp = reg_XBGR.get_booster().get_score(importance_type=param_imp)

    # obtém nome das colunas
    keys = list(reg_XBGR_feature_imp.keys())

    # obtém scores das features
    values = list(reg_XBGR_feature_imp.values())

    # crianndo dataframe  com  k recusros principais
    xbg_best_features = pd.DataFrame(data=values, index=keys, columns=["score_XGBRFRegressor"]).sort_values(
        by="score_XGBRFRegressor", ascending=True).nlargest(n_best_features, columns="score_XGBRFRegressor")

    # Return the best features
    return xbg_best_features

# Creating a function to make things easier selecting model parameters XGB
def xgb_model_helper(X_train, y_train, PARAMETERS, V_PARAM_NAME=False, V_PARAM_VALUES=False, BR=10,):
    # Cria uma matrix temporária em formato de bit do conjunto de dados a ser treinados
    temp_dmatrix = xgb.DMatrix(data=X_train, label=y_train)

    # Check os parâmetros a ser utilizados
    if V_PARAM_VALUES == False:
        cv_results = xgb.cv(dtrain=temp_dmatrix, nfold=5, num_boost_round=BR, params=PARAMETERS, as_pandas=True,
                            seed=123)
        return cv_results

    else:
        # Criando uma Lista, para armazenar os resultados e os nomes, de cada uma das métricas.
        results = []

        # Percorre a lista de parâmetros
        for v_param_value in V_PARAM_VALUES:
            # Adicionando o nome dos parâmetros avaliado a lista de nomes.
            PARAMETERS[V_PARAM_NAME] = v_param_value

            # Treinando o modelo com Cross Validation.
            cv_results = xgb.cv(dtrain=temp_dmatrix, nfold=5, num_boost_round=BR, params=PARAMETERS, as_pandas=True,
                                seed=123)

            # Adicionando os resultados gerados a lista de resultados.
            results.append((cv_results["train-mae-mean"].tail().values[-1], cv_results["test-mae-mean"].tail().values[
                -1]))  # .tail().values[-1] captura somente as colunas

        # zip “pareia” os elementos de uma série de listas, tuplas ou outras sequências para criar uma lista de tuplas:

        # Adicionando a média da AUC e o desvio-padrão dos resultados gerados, pelo modelo analisado ao Dataframe de médias.
        data = list(zip(V_PARAM_VALUES, results))
        print(pd.DataFrame(data, columns=[V_PARAM_NAME, "mae"]))

        return cv_results

# Creating a function for Optimizing the number of reinforcement rounds (since I used xgb's DMAtrix)
def opt_number_of_boosting_rounds(X_train, y_train,):
    # create the DMatrix
    temp_dmatrix = xgb.DMatrix(data=X_train, label=y_train)

    # Create the parameter dictionary for each tree: params
    params = {"objective": 'reg:linear', "max_depth": 5}

    # Create lis of number of boosting rounds
    num_rounds = [5, 10, 20, 25, 50, 100]

    # Empty list to store final round rmse per XGBoost model
    final_rmse_per_round = []

    # Iterate ove num_rounds and build one model per num_boost_round parameter
    for curr_num_rounds in num_rounds:
        # Perform cross-validation: cv_results
        cv_results = xgb.cv(dtrain=temp_dmatrix, params=params, nfold=5, num_boost_round=curr_num_rounds,
                            metrics="mae", as_pandas=True, seed=123)
        # Append final round RMSE
        final_rmse_per_round.append(cv_results["test-mae-mean"].tail().values[-1])
    # print the resultant Dataframe
    num_rounds_rmses = list(zip(num_rounds, final_rmse_per_round))

    return pd.DataFrame(num_rounds_rmses, columns=["num_boosting_rounds", "mae"])
#-------------------------------------------------------

# Deep Learning

""""A named constant is a name that represents a value that cannot be 
changed during the program's execution."""

# Using Global Constants Defining Named Constants
SEQ_LENGTH = 84
BATCH_SIZE = 32
#-------------------------------------------------------

# Creating an function for Instantiating datasets for training, validation, and testing
def to_windows(dataset, length):
    dataset = dataset.window(length, shift=1, drop_remainder=True)
    return dataset.flat_map(lambda window_ds: window_ds.batch(length))


def to_seq2seq_dataset(series, ahead, seq_length=SEQ_LENGTH, target_cols=1, batch_size=BATCH_SIZE,
                       shuffle=False, seed=None):
    ds = to_windows(tf.data.Dataset.from_tensor_slices(series), ahead + 1)
    ds = to_windows(ds, seq_length).map(lambda S: (S[:, 0], S[:, 1:, 1]))
    if shuffle:
        ds = ds.shuffle(8 * batch_size, seed=seed)
    return ds.batch(batch_size)

# Creating a function for TensorBoard
# Setting log path
root_logdir = os.path.join(os.curdir, "world_comics_logs")

def get_run_logdir(root_logdir="logs"):
    return Path(root_logdir) / strftime("run_%Y_%m_%d_%H_%M_%S")

def name_logdir(name):
    # Increment every time you train the model
    run_index = 1
    run_logdir = Path() / "world_comics_logs" / f"{name}"

    # Return Callback TensorBoard()
    return tf.keras.callbacks.TensorBoard(run_logdir)

# Create an function for Model training
def fit_and_evaluate(model, train_set, valid_set, learning_rate, filepath, logdir, epochs=None):
    # Create an list of callbacks

    # We'll use a callback to stop training when our performance metric reaches a specified level.
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_mae",
        # mode='min',
        patience=50,
        restore_best_weights=True,
    )
    # We use a callback to save the best performing model.
    model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        save_weights_only=True,
        monitor='val_mae',
        # mode='min',
        save_best_only=True,
    )
    # To define a List of callbacks to use
    callbacks = [early_stopping_cb, model_checkpoint_cb, name_logdir(logdir), TqdmCallback(verbose=0)]

    # Compile the model
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate),
                  metrics=["mae", "mape", "mse", "msle"]
                  )
    # Fit the model
    history = model.fit(train_set, validation_data=valid_set, epochs=epochs,
                        verbose=0,
                        callbacks=callbacks
                        )
    # Evaluate
    # zip “pairs” elements from a series of lists, tuples, or other sequences to create a list of tuples:
    data = list(zip(model.metrics_names, model.evaluate(valid_set)))

    # Returns dataframe with values of metrics calculated by the model
    return pd.DataFrame(data, columns=['Metrics', 'Score'])

# Creating a function to evaluate the model
def predictions(model, dataset_valid, dataset_test, target,
                seq2seq_valid, seq2seq_test, ahead, seq_length=SEQ_LENGTH):
    # Evaluation Predictions
    print('Forecast in Data Validation')
    for i in range(ahead):
        preds = pd.Series(model.predict(seq2seq_valid)[:-1, -1, i],
                          index=dataset_valid.index[seq_length + i: -ahead + i])
        mae = (preds - dataset_valid[target]).abs().mean() * 1e6
        print(f"MAE for + {i + 1}: {mae:,.0f}")
    print('\n')

    print('Forecast in Data Test')
    for i in range(ahead):
        preds = pd.Series(model.predict(seq2seq_test)[:-1, -1, i],
                          index=dataset_test.index[seq_length + i: -ahead + i])
        mae = (preds - dataset_test[target]).abs().mean() * 1e6
        print(f"MAE for +{i + 1}: {mae:,.0f}")