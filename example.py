import json
from DiscriminationMitigation import *
from sklearn.model_selection import train_test_split
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 100)

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


if __name__ == '__main__':

    synth = simple_synth()
    synth['z'] = np.random.randint(low=1, high=5, size=len(synth))

    with open('config.json') as j:
        config = json.load(j)

    with open('weights.json') as j:
        weights = json.load(j)

    # Train (and val) / test split
    X_train, X_test, y_train, y_test = train_test_split(synth.loc[:, ~synth.columns.isin(config['target_feature'])],
                                                        synth[config['target_feature']], random_state=123,
                                                        test_size=500)
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

    # Tensorflow Keras Model class
    tf.keras.backend.clear_session()
    inputs = tf.keras.layers.Input(shape=4,)
    dense = tf.keras.layers.Dense(8)(inputs)
    dropout = tf.keras.layers.Dropout(0.3)(dense)
    dense = tf.keras.layers.Dense(16)(dropout)
    dropout = tf.keras.layers.Dropout(0.1)(dense)
    output = tf.keras.layers.Dense(1, activation='linear', name='output')(dropout)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=80, batch_size=64, validation_data=(X_val, y_val))

    dm = DiscriminationMitigator(df=X_test, model=model, config=config, train=X_train, weights=weights)
    predictions = dm.predictions()