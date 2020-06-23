import json
from DiscriminationMitigation import *
from sklearn.model_selection import train_test_split
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 100)

from importlib import reload
import DiscriminationMitigation.DiscriminationMitigator
reload(DiscriminationMitigation.DiscriminationMitigator)

def simple_synth(n=10000, class_probab=0.5, gamma0=4, gamma1=6, alpha0=2, alpha1=1, beta0=1, beta1=1):

    np.random.seed(123)

    # Protected class variable
    c1 = np.random.binomial(1, p=class_probab, size=n) # group 1
    c0 = 1-c1 # group 0

    # Other covariate
    w = gamma0*c0 + gamma1*c1 + np.random.normal(0, 0.5, size=n) # linear function of class & shock

    # Outcome variable
    y = alpha0*c0 + alpha1*c1 + beta0*c0*w + beta1*c1*w + np.random.normal(0, 0.5, size=n)

    return pd.DataFrame([y, c0, c1, w]).T.rename(columns={0:'y', 1: 'c0', 2: 'c1', 3: 'w'})

def convert_lists(df, categorical_features, numeric_features):
    return [df[col] for col in categorical_features] + [df[numeric_features]]


if __name__ == '__main__':

    synth = simple_synth()
    synth['z'] = np.random.randint(low=1, high=5, size=len(synth))

    with open('example_config.json') as j:
        config = json.load(j)

    with open('example_weights.json') as j:
        weights = json.load(j)

    synth['a'] = np.random.randint(low=1, high=2, size=len(synth))
    synth['b'] = np.random.randint(low=1, high=15, size=len(synth))
    synth['c'] = np.random.randint(low=5, high=20, size=len(synth))

    # Train (and val) / test split
    X_train, X_test, y_train, y_test = train_test_split(synth.loc[:, ~synth.columns.isin(config['target_feature'])],
                                                        synth[config['target_feature']], random_state=123,
                                                        test_size=500)
    # Train / val split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=123, test_size=0.2)

    categorical_features = config['protected_class_features']
    numeric_features = [i for i in synth.columns if i not in categorical_features and i not in config['target_feature']]

    X_train = convert_lists(X_train, categorical_features, numeric_features)
    X_test = convert_lists(X_test, categorical_features, numeric_features)
    X_val = convert_lists(X_val, categorical_features, numeric_features)

    # Tensorflow Keras Model class
    tf.keras.backend.clear_session()
    inputs = tf.keras.layers.Input(shape=1,)
    dense = tf.keras.layers.Dense(8)(inputs)
    dropout = tf.keras.layers.Dropout(0.3)(dense)
    dense = tf.keras.layers.Dense(16)(dropout)
    dropout = tf.keras.layers.Dropout(0.1)(dense)
    output = tf.keras.layers.Dense(1, activation='linear', name='output')(dropout)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_val, y_val))

    pred = DiscriminationMitigator(df=X_test, model=model, config=config).predictions()