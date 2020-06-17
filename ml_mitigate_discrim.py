import json, warnings
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

    def iterate_predictions(self):
        '''
        Method iteratively generates N x 1 vector of predictions across all unique categorical value (K) of all
            protected class features (C). On each iteration, all observations in self.df are (re)assigned the value Ki (i.e.
            a particular categorical value of feature c) and using this altered dataframe the model generates an N x 1
            vector of predictions. The final output dataframe is N x K dimensions.
        :return: Pandas dataframe
        '''
        predictions = pd.DataFrame()
        for feature in self.config['protected_class_features']:
            for val in self.df[feature].unique():
                temp = self.df.copy()
                temp[feature] = val
                predictions = pd.concat([predictions, pd.DataFrame(model.predict(temp),index=self.df.index).rename(
                    columns={0: feature + '_' + str(val)})], axis=1)
        return predictions

    def feature_marginals(self, df):
        '''
        Generates dictionary of marginal distributions per feature in a dataframe.
        :param df: Pandas dataframe, input dataframe from which marginal distributions for each protected class feature generated.
        :return: dictionary of marginal distributions, e.g. {'feature1': {categ1: 0.8, categ2: 0.2}, 'feature2'...} of `df`
        '''
        marginals = {}
        for feature in self.config['protected_class_features']:
            marginals[feature] = df[feature].value_counts(normalize=True, dropna=False).to_dict()
        return marginals

    def check_for_onehot(self):
        '''
        Checks whether any protected class features in self.df are extremely correlated, suggesting one-hot vectors. Users
            must ensure that if there is no reference category in trained model that adjacent one-hot vector marginals
            uniquely identify observations. E.g. for two one-hot vectors, if 80% of observations for vector1=1, vector2=1
            must be the corresponding 20%.
        :return: nothing, a warning message if two or more features are correlated > 0.9999
        '''
        correlations = self.df.corr().abs().iloc[0, :]  # correlation matrix, selecting first row
        extreme_corr = correlations[correlations > 0.9999].index.tolist()

        if extreme_corr:
            warnings.warn("\n Warning! The following features are extremely correlated and thus may be one-hot vectors: {}. \n"
                          "If no category is omitted, users must ensure custom marginal weights for one-hot vectors align correctly.".format(' '.join(extreme_corr)))

    def weighted_predictions(self, marginal_dict, prediction_df):
        '''
        Method weights the N x K matrix of predictions in `prediction_df` according to the marginal distribution of each
            particular categorical value, Ki, based on dictionary of marginals. The reduces the dimension of the prediction
            matrix down to N x C, where each column is the weighted average (from `marginal_dict`) of protected class
            feature c.
        :param marginal_dict: dictionary of marginal distributions per feature in input dataframe
        :param prediction_df: Pandas dataframe, iterated predictions from self.iterate_predictions()
        :return: Pandas dataframe with columns being the weighted predictions for each protected class feature, c.
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

    def predictions(self):
        '''
        Generates predictions by calling methods of class.
        :return: Pandas dataframe of 3 or possibly 4 columns of predictions:
            'unadj_pred' - unadjusted predictions for self.df
            'unif_wts' - predictions with uniform weights (i.e. simple average across N x K matrix of predictions)
            'pop_wts' - predictions weighted to reflect the marginal distribution in the training set (if provided) otherwise self.df
                optionally:
                'cust_wts' - predictions with user-specified marginal weights.
        '''

        # N x K matrix of predictions
        iterated_predictions = self.iterate_predictions()

        # Marginals for either train or df
        if self.train is not None:
            marginals = self.feature_marginals(
                self.train)  # marginals from training set to reweight test with same composition
        else:
            marginals = self.feature_marginals(self.df)

        # Collapse down to N x C matrix of weighted predictions
        weighted_predictions = self.weighted_predictions(marginals, iterated_predictions)

        # Output dataframe of adjusted final predictions
        output_predictions = pd.DataFrame()
        output_predictions['unadj_pred'] = pd.DataFrame(self.model.predict(self.df), index=self.df.index)[0] # unadjusted predictions
        output_predictions['unif_wts'] = iterated_predictions.mean(axis=1)  # uniform weights (i.e. simple average)
        output_predictions['pop_wts'] = weighted_predictions.mean(axis=1)  # weighted to match train or other marginal dist

        # Dictionary of custom weights that combine user-supplied weights with marginals of either train or df
        if self.weights is not None:
            custom_weights = {}
            for feature in marginals.keys():
                if feature in self.weights.keys():
                    custom_weights[feature] = self.weights[feature]
                else:
                    custom_weights[feature] = marginals[feature]

            self.check_for_onehot() # check if one-hot vectors possibly present and warn

            # Condense user-specified marginal weights to N x 1 vector
            reweighted_predictions = self.weighted_predictions(custom_weights, iterated_predictions)
            output_predictions['cust_wts'] = reweighted_predictions.mean(axis=1)

        return output_predictions


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

    dm = DiscriminationMitigator(df=X_test, model=model, config=config)
    predictions = dm.predictions()