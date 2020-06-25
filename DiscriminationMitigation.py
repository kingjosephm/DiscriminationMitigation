import warnings, copy
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import List, Dict, Union

def custom_formatwarning(msg, *args, **kwargs):
    return str(msg) + '\n'
warnings.formatwarning = custom_formatwarning

class DiscriminationMitigator:

    def __init__(self,
                 df: Union[List[Union[pd.core.series.Series, pd.core.frame.DataFrame]], pd.core.frame.DataFrame],
                 model: Union[tf.python.keras.engine.sequential.Sequential, tf.python.keras.engine.training.Model],
                 config: Dict,
                 train: Union[None, List[Union[pd.core.series.Series, pd.core.frame.DataFrame]], pd.core.frame.DataFrame] = None,
                 weights: Union[None, Dict] = None) -> None:
        self.df = df
        self.model = model
        self.config = config
        self.train = train
        self.weights = weights

        # Ensure all protected class features listed in config also in df
        if not all(elem in self.ensure_dataframe(self.df).columns for elem in self.config['protected_class_features']):
            raise ValueError("\nPlease ensure all protected class features are in parameter df!")

        if self.weights is not None:
            self.weights = self.check_weights(weights)

    def check_weights(self, weights: Dict) -> Dict:
        '''
        Changes weight category codes (keys) from strings (JSON required) to floats. Also ensures marginals
        per feature sum to 1.
        :param weights: dict, user-supplied custom marginal distributions
        :return: modified dictionary of weights
        '''
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
                raise ValueError("\nThe marginals for feature '{}' do not sum to 1! Marginals must sum to 1!"\
                    .format(feature_name))
        return reweights

    def ensure_dataframe(self, df: Union[List[Union[pd.core.series.Series, pd.core.frame.DataFrame]], pd.core.frame.DataFrame]) -> pd.core.frame.DataFrame:
        '''
        Ensures df is a pd.DataFrame. If list, converts to pd.DataFrame, else simply returns pd.DataFrame
        :param df: list or pd.DataFrame
        :return: pd.DataFrame
        '''

        if isinstance(df, list):
            for i in range(len(df)):
                if i == 0:
                    dataframe = df[0]
                else:
                    dataframe = pd.concat([df[i], dataframe], axis=1)
        else:
            dataframe = df
        return dataframe

    def iterate_predictions(self) -> pd.core.frame.DataFrame:
        '''
        Method iteratively generates N x 1 vector of predictions across all unique categorical value (K) of all
            protected class features (C). On each iteration, all observations in self.df are (re)assigned the value Ki (i.e.
            a particular categorical value of feature c) and using this altered dataframe the model generates an N x 1
            vector of predictions. The final output dataframe is N x K dimensions.
        :return: Pandas dataframe
        '''
        predictions = pd.DataFrame()
        if isinstance(self.df, pd.core.frame.DataFrame):  # if self.df pd.DataFrame
            for feature in self.config['protected_class_features']:
                for val in self.df[feature].unique():
                    temp = copy.deepcopy(self.df)
                    temp[feature] = val
                    predictions = pd.concat([predictions, pd.DataFrame(self.model.predict(temp), index=self.df.index).rename(
                        columns={0: feature + '_' + str(float(val))})], axis=1)
        else:  # if self.df list
            for feature in self.config['protected_class_features']:
                for i in range(len(self.df)):
                    if isinstance(self.df[i],
                                  pd.core.series.Series):  # syntax for column name of pd.Series different than pd.DataFrame
                        if self.df[i].name == feature:
                            for val in self.df[i].unique():
                                temp = copy.deepcopy(self.df)
                                temp[i].values[:] = val
                                predictions = pd.concat(
                                    [predictions, pd.DataFrame(self.model.predict(temp), index=self.df[i].index).rename(
                                        columns={0: feature + '_' + str(float(val))})], axis=1)
                        else:
                            pass  # individual feature is not a protected class feature
                    else:  # if pd.DataFrame within list
                        if [j for j in self.df[i].columns if feature in j]:  # check if feature in dataframe
                            for val in self.df[i][feature].unique():
                                temp = copy.deepcopy(self.df)
                                temp[i][feature] = val
                                predictions = pd.concat(
                                    [predictions, pd.DataFrame(self.model.predict(temp), index=self.df[i].index).rename(
                                        columns={0: feature + '_' + str(float(val))})], axis=1)
                        else:
                            pass  # no protected class feature(s) in dataframe
        return predictions

    def unadjusted_prediction(self) -> pd.core.series.Series:
        '''
        Estimates unadjusted model predictions, with syntax varying slightly depending on input data type of self.df
        :return: pd.Series of unadjusted predictions
        '''
        if isinstance(self.df, pd.core.frame.DataFrame):
            unadj_pred = pd.DataFrame(self.model.predict(self.df), index=self.df.index)[0]  # unadjusted predictions
        else: # if list
            unadj_pred = pd.DataFrame(self.model.predict(self.df), index=self.df[0].index)[0]  # unadjusted predictions
        return unadj_pred

    def feature_marginals(self, df: pd.core.frame.DataFrame) -> Dict:
        '''
        Generates dictionary of marginal distributions per feature in a dataframe.
        :param df: Pandas dataframe, input dataframe from which marginal distributions for each protected class feature generated.
        :return: dictionary of marginal distributions, e.g. {'feature1': {categ1: 0.8, categ2: 0.2}, 'feature2'...} of `df`
        '''
        marginals = {}
        for feature in self.config['protected_class_features']:
            marginals[feature] = df[feature].value_counts(normalize=True, dropna=False).to_dict()
        return marginals

    def adjust_missing_categ_vals(self, dictionary: Dict, feature: str, iterated_predictions: pd.core.frame.DataFrame) -> Dict:
        '''
        Method checks whether all categorial values (keys) present in dictionary are also in iterated_predictions (i.e. self.df)
            and consolidates values corresponding to missing keys into overlapping keys between two sources
        :param dictionary: dict, inner part of nested dictionary from self.feature_marginals where keys are values of a given protected class feature
        :param feature: str, protected class feature name
        :param iterated_predictions: Pandas dataframe, N x K matrix of predictions from self.iterate_predictions
        :return: dictionary of overlapping keys between param dictionary and parm iterated_predictions
        '''

        # Identify which if any category values (key(s)) of feature missing
        dict_keys = list(dictionary.keys())
        feature_values = [float(i.split('_')[-1]) for i in iterated_predictions.columns if feature + '_' in i]
        missing_keys = [i for i in dict_keys if i not in feature_values]

        if missing_keys:
            warnings.warn("\nThe following category value(s) of feature '{}' are present in test dataframe, but not in \n"
                  "df supplied: {}. These values cannot be directly reweight in supplied df. Weights of \n"
                  "overlapping categories in both dataframes will be adjusted. \n"\
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

    def check_for_onehot(self) -> None:
        '''
        Checks whether any protected class features in self.df are extremely correlated, suggesting one-hot vectors. Users
            must ensure that if there is no reference category in trained model that adjacent one-hot vector marginals
            uniquely identify observations. E.g. for two one-hot vectors, if 80% of observations for vector1=1, vector2=1
            must be the corresponding 20%.
        :return: nothing, a warning message if two or more features are correlated > 0.9999
        '''
        correlations = self.ensure_dataframe(self.df).corr().abs().iloc[0, :]  # correlation matrix, selecting first row
        extreme_corr = correlations[correlations > 0.9999].index.tolist()

        if extreme_corr:
            warnings.warn("\nWarning! The following features are extremely correlated and thus may be one-hot vectors: {}. \n"
                 "If no category is omitted, users must ensure custom marginal weights for one-hot vectors align correctly.".format(' '.join(extreme_corr)))

    def weighted_predictions(self, marginal_dict: Dict, prediction_df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
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

    def predictions(self) -> pd.core.frame.DataFrame:
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
            marginals = self.feature_marginals(self.ensure_dataframe(self.train))  # marginals from training set to reweight test with same composition
            for feature in marginals.keys(): # ensure all categorical values for each feature present in test also in self.df
                marginals[feature].update(self.adjust_missing_categ_vals(marginals[feature], feature, iterated_predictions))
        else:
            marginals = self.feature_marginals(self.ensure_dataframe(self.df))

        # Collapse down to N x C matrix of weighted predictions
        weighted_predictions = self.weighted_predictions(marginals, iterated_predictions)

        # Output dataframe of adjusted final predictions
        output_predictions = pd.DataFrame()
        output_predictions['unadj_pred'] = self.unadjusted_prediction()
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