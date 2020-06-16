import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils import *


class DiscriminationMitigator:

    def __init__(self, df, model, config, train=None, weights=None):
        self.df = df
        self.model = model
        self.config = config
        self.train = train
        self.weights = weights

        # Ensure inputs are correct type
        assert (isinstance(self.df, pd.DataFrame)), "\nPlease ensure parameter df is a Pandas dataframe!"
        assert (isinstance(self.model, (tf.python.keras.engine.sequential.Sequential,
                                        tf.python.keras.engine.training.Model))), "\nPlease parameter model is either tf.keras Model or Sequential class!"
        assert (isinstance(config, dict)), "\nPlease ensure parameter config is a dictionary!"

        # Ensure all protected class features in data
        if not all(elem in self.df.columns for elem in self.config['protected_class_features']):
            raise ValueError("\nPlease ensure all protected class features are in parameter df!")

        if self.train is not None:
            assert (isinstance(self.train, pd.DataFrame))

        if self.weights is not None:
            assert (isinstance(self.weights, dict))

            # Change weight category codes from str -> float (json requires keys as string)
            # Also ensure marginals per feature sum to 1
            reweights = {}
            for feature_name in weights.keys():  # individual protected feature
                marginal_sum = 0
                for categ_value, share in weights[
                    feature_name].items():  # category value associated with a protected class feature
                    if feature_name not in reweights:
                        reweights[feature_name] = {}
                    reweights[feature_name].update({float(categ_value): share})
                    marginal_sum += share
                if marginal_sum != 1.0:
                    raise ValueError(
                        "\nThe marginal sum for feature '{}' does not sum to 1! Marginals must sum to 1!".format(
                            feature_name))
            self.weights = reweights

    def unadjusted_prediction(self):
        '''
        Estimate unadjusted fitted values using input dataframe, df
        :return: Pandas Series
        '''
        return pd.DataFrame(self.model.predict(self.df), index=self.df.index)[0]

    def iterate_predictions(self):
        '''
        Method iterates through all values of all protected class features, assigning on each iteration all observations
            the value of that particular protected class.
        :return: Pandas dataframe
        '''
        predictions = pd.DataFrame()
        for feature in self.config['protected_class_features']:
            for val in self.df[feature].unique():
                temp = self.df.copy()
                temp[feature] = val
                predictions = pd.concat([predictions, pd.DataFrame(model.predict(temp),
                                                                   index=self.df.index).rename(
                    columns={0: feature + '_' + str(val)})], axis=1)
        return predictions

    def feature_marginals(self, df):
        '''
        Generates dictionary of marginal distributions per feature in a dataframe
        :param df: Pandas dataframe
        :return: dictionary
        '''
        marginals = {}
        for feature in self.config['protected_class_features']:
            marginals[feature] = df[feature].value_counts(normalize=True, dropna=False).to_dict()
        return marginals

    def weighted_predictions(self, marginal_dict, prediction_df):
        '''
        Weights predictions for each feature-value by marginal distribution of that value in test set (if provided)
            otherwise weights by marginal distribution of df supplied
        :param marginal_dict: dictionary of marginal distributions per feature in input dataframe
        :param prediction_df: Pandas dataframe, iterated predictions from self.iterate_predictions()
        :return: Pandas dataframe with columns being the weighted predictions for each protected class feature
        '''
        wt_pred = pd.DataFrame()
        for feature, val in marginal_dict.items():
            wt = np.zeros(shape=len(prediction_df))
            if isinstance(val, dict):  # check if nested dictionary, must be if 1+ value per feature
                for elem, share in marginal_dict[feature].items():
                    wt += prediction_df[feature + '_' + str(elem)] * share
            else:  # if feature invariant
                raise ValueError("\nProtected class feature '{}' is invariant!".format(feature))
            wt_pred = pd.concat([wt_pred, pd.DataFrame({feature: wt})], axis=1)
        return wt_pred

    def adjust_predictions(self):

        predictions = self.iterate_predictions()

        if self.train is not None:
            marginals = self.feature_marginals(
                self.train)  # marginals from training set to reweight test with same composition
        else:
            marginals = self.feature_marginals(self.df)

        weighted_predictions = self.weighted_predictions(marginals, predictions)

        adjusted_predictions = pd.DataFrame()
        adjusted_predictions['unif_wts'] = predictions.mean(axis=1)  # uniform weights (i.e. simple average)
        adjusted_predictions['pop_wts'] = weighted_predictions.mean(
            axis=1)  # weighted to match train or other marginal dist

        return adjusted_predictions


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
    inputs = tf.keras.Input(shape=3,)
    dense = tf.keras.layers.Dense(8)(inputs)
    dropout = tf.keras.layers.Dropout(0.3)(dense)
    dense = tf.keras.layers.Dense(16)(dropout)
    dropout = tf.keras.layers.Dropout(0.1)(dense)
    output = tf.keras.layers.Dense(1, activation='linear', name='output')(dropout)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=80, batch_size=64, validation_data=(X_val, y_val))

    dm = DiscriminationMitigator(df=X_test, model=model, config=config, train=X_train, weights=weights)
    dm.feature_marginals(dm.df)
    adjusted_predictions = dm.adjust_predictions()