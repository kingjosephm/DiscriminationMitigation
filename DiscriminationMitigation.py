import warnings
import pandas as pd
import numpy as np
import tensorflow as tf

class DiscriminationMitigator:

    def __init__(self, df, model, config, train=None, weights=None):
        self.df = df
        self.model = model
        self.config = config
        self.train = train
        self.weights = weights

        # Ensure inputs are correct type and format
        assert (isinstance(self.df, pd.DataFrame)), "\nPlease ensure parameter df is a Pandas dataframe!"

        assert(self.df.shape[1] == self.df.select_dtypes(include=np.number).shape[1]), "\nDataframe supplied in param df must be all numeric dtypes!"

        assert (isinstance(self.model, (tf.python.keras.engine.sequential.Sequential,
                                        tf.python.keras.engine.training.Model))), "\nPlease verify parameter model is either tf.keras Model or Sequential class!"

        assert (isinstance(config, dict)), "\nPlease ensure parameter config is a dictionary!"

        # Ensure all protected class features in data
        if not all(elem in self.df.columns for elem in self.config['protected_class_features']):
            raise ValueError("\nPlease ensure all protected class features are in parameter df!")

        if self.train is not None:
            assert (isinstance(self.train, pd.DataFrame))

            assert (self.train.shape[1] == self.train.select_dtypes(include=np.number).shape[1]), "\nDataframe supplied in param train must be all numeric dtypes!"

            if not all(col in self.df.columns for col in self.train.columns):
                raise KeyError("\nNot all columns in df are in train! Please ensure they are and try again.")

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
                        "\nThe marginals for feature '{}' do not sum to 1! Marginals must sum to 1!".format(
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
                predictions = pd.concat([predictions, pd.DataFrame(self.model.predict(temp), index=self.df.index).rename(
                    columns={0: feature + '_' + str(float(val))})], axis=1)
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

    def adjust_missing_categ_vals(sel, dictionary, feature, iterated_predictions):
        '''
        Method checks whether all categorial values (keys) present in dictionary are also in iterated_predictions (i.e. self.df)
            and consolidates values corresponding to missing keys into overlapping keys between two sources
        :param dictionary: dict, inner part of nested dictionary from self.feature_marginals where keys are values of a given protected class feature
        :param feature: str, protected class feature name
        :param iterated_predictions: Pandas dataframe, N x K matrix of predictions from self.iterate_predictions
        :return:
        '''
        # Identify which if any category values (key(s)) of feature missing
        dict_keys = list(dictionary.keys())
        feature_values = [float(i.split('_')[-1]) for i in iterated_predictions.columns if feature + '_' in i]
        missing_keys = [i for i in dict_keys if i not in feature_values]

        if missing_keys:
            warnings.warn("\nThe following category value(s) of feature '{}' are present in test dataframe, but not in \n"
                          "df supplied: {}. These values cannot be directly reweight in supplied df. Weights of \n"
                          "overlapping categories in both dataframes will be adjusted."\
                          .format(feature, ' '.join([str(i) for i in missing_keys])))

        # Sum share of all categories not present in self.df but in self.train
        tot_share = 0
        for x in missing_keys:
            tot_share += dictionary[x]
        adjustment = 1 - tot_share

        # Adjust overlapping keys
        overlapping_keys = [i for i in dict_keys if i not in missing_keys]
        for key in overlapping_keys:
            dictionary[key] = dictionary[key] / adjustment

        # Delete non-overlapping keys from dictionary
        for key in missing_keys:
            del dictionary[key]

        return dictionary

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
                    try:
                        wt += prediction_df[feature + '_' + str(float(elem))] * share
                    except KeyError: # catch if key not part of dictionary
                        raise Exception("\nThe category value '{}' in feature '{}' of supplied dictionary does not exist \n"
                              "in supplied dataframe! Please ensure all values for all protected class features \n"
                              "in this dictionary exist in the data.".format(elem, feature))
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
            marginals = self.feature_marginals(self.train)  # marginals from training set to reweight test with same composition
            for feature in marginals.keys(): # ensure all categorical values for each feature present in test also in self.df
                marginals[feature].update(self.adjust_missing_categ_vals(marginals[feature], feature, iterated_predictions))
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