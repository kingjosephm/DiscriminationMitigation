import pandas as pd
import numpy as np
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def synth_data(n=10000, class_probab=0.5, gamma0=4, gamma1=6, alpha0=2, alpha1=1, beta0=1, beta1=1):

    np.random.seed(123)

    # Protected class variable
    c1 = np.random.binomial(1, p=class_probab, size=n) # group 1
    c0 = 1-c1 # group 0

    # Other covariate
    w = gamma0*c0 + gamma1*c1 + np.random.normal(0, 0.5, size=n) # linear function of class & shock

    # Outcome variable
    y = alpha0*c0 + alpha1*c1 + beta0*c0*w + beta1*c1*w + np.random.normal(0, 0.5, size=n)

    return pd.DataFrame([y, c0, c1, w]).T.rename(columns={0:'y', 1: 'c0', 2: 'c1', 3: 'w'})

def make_plot(constrain_N=5):
    val_temp = model.history.history['val_loss'].copy()
    train_temp = model.history.history['loss'].copy()


    for i in range(min(len(train_temp), constrain_N)):
        train_temp[i] = np.NaN
        val_temp[i] = np.NaN

    plt.clf()
    plt.close()
    plt.plot(train_temp)
    plt.plot(val_temp)
    plt.title('Validation loss')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'])

def some_function(df, model, config):

    # Ensure all protected class features in data
    if not all(elem in df.columns for elem in config['protected_class_features']):
        raise ValueError("\nNot all protected class features found in data! Verify config file and try again.")

    # Ensure correct model type (we'll have to expand this later)
    assert(isinstance(model, (tf.python.keras.engine.sequential.Sequential,
                              tf.python.keras.engine.training.Model))), \
    "\nPlease verify model is either tf.keras Model or Sequential class!"

    predictions = pd.DataFrame(model.predict(df), index=df.index).rename(columns={0: 'pred_unadj'})
    for feature in config['protected_class_features']:
        for val in df[feature].unique():
            temp = df.copy() # df to recode
            temp[feature] = val
            predictions = pd.concat([predictions, pd.DataFrame(model.predict(temp),
                          index=df.index).rename(columns={0: feature + '_' + str(int(val))})], axis=1)

    predictions['simp_avg'] = predictions.iloc[:, 1:].mean(axis=1) # simple average across predictions

    return predictions

if __name__ == '__main__':

    df = synth_data()

    with open('config.json') as j:
        config = json.load(j)

    # Train (and val) / test split
    X_train, X_test, y_train, y_test = train_test_split(df.loc[:, ~df.columns.isin(config['target_feature'])],
                df[config['target_feature']], random_state=123, test_size=500)
    # Train / val split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=123, test_size=0.2)


    # Tensorflow Keras Sequential class
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(8))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(16))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=80, batch_size=64, validation_data=(X_val, y_val))
    #make_plot(constrain_N=20)

    # Tensorflow Keras Model class
    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=3,)
    dense = tf.keras.layers.Dense(8)(inputs)
    dropout = tf.keras.layers.Dropout(0.3)(dense)
    dense = tf.keras.layers.Dense(16)(dropout)
    dropout = tf.keras.layers.Dropout(0.1)(dense)
    output = tf.keras.layers.Dense(1, activation='linear', name='output')(dropout)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=80, batch_size=64, validation_data=(X_val, y_val))

    predictions = some_function(X_test, model, config)

