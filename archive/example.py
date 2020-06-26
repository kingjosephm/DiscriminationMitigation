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

    #X_train = [X_train]
    #X_test = [X_test]
    #X_val = [X_val]

    # Tensorflow Keras Sequential class
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=7,))
    model.add(tf.keras.layers.Dense(8))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(16))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_val, y_val))

    dm = DiscriminationMitigator(df=X_test, model=model, config=config, train=X_train, weights=weights)
    pred = dm.predictions()




    iterated_predictions = dm.iterate_predictions()
    marginals = dm.feature_marginals(dm.ensure_dataframe((dm.train)))
    for feature in marginals.keys():
        marginals[feature].update(dm.adjust_missing_categ_vals(marginals[feature], feature, iterated_predictions))

    reweighted_predictions = dm.weighted_predictions(dm.weights, iterated_predictions)
    weighted_predictions = dm.weighted_predictions(marginals, iterated_predictions)

    foo = pd.DataFrame()
    pd.set_option('precision', 8)
    foo['pop'] = dm.weighted_predictions(marginals, iterated_predictions).mean(axis=1)
    foo['cust'] = dm.weighted_predictions(dm.weights, iterated_predictions).mean(axis=1)
    foo.corr()
    foo.describe()

    custom_weights = {'c0': {0.0: 0.1, 1.0: 0.9},
        'c1': {0.0: 0.9, 1.0: 0.1},
        'z': {1.0: 0.9, 2.0: 0.02, 3.0: 0.04, 4.0: 0.04}}


    def weighted_predictions(marginal_dict, prediction_df):
        wt_pred = pd.DataFrame()
        for feature, val in marginal_dict.items():
            wt = np.zeros(shape=len(prediction_df))
            if isinstance(val, dict):  # check if nested dictionary, must be if 1+ value per feature
                for elem, share in marginal_dict[feature].items():
                    try:
                        wt += prediction_df[feature + '_' + str(float(elem))] * share
                    except KeyError:  # catch if key not part of dictionary
                        raise Exception("\nThe category value '{}' in feature '{}' of supplied dictionary does not exist \n"
                                        "in supplied dataframe! Please ensure all values for all protected class features \n"
                                        "in this dictionary exist in the data.".format(elem, feature))
            else:  # if feature invariant
                raise ValueError("\nProtected class feature '{}' is invariant!".format(feature))
            wt_pred = pd.concat([wt_pred, pd.DataFrame({feature: wt})], axis=1)
        return wt_pred

    one = weighted_predictions(custom_weights, iterated_predictions)
    two = weighted_predictions(marginals, iterated_predictions)
    comb = pd.concat([one, two], axis=1)