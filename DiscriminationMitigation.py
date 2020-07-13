import warnings, copy, lightgbm
import pandas as pd
import numpy as np
import tensorflow as tf
from itertools import product
from typing import List, Dict, Union, Tuple

def custom_formatwarning(msg, *args, **kwargs):
    return str(msg) + '\n'
warnings.formatwarning = custom_formatwarning

class DiscriminationMitigator:

    def __init__(self,
                 df: Union[List[Union[pd.core.series.Series, pd.core.frame.DataFrame]], pd.core.frame.DataFrame],
                 model: Union[tf.python.keras.engine.sequential.Sequential, tf.python.keras.engine.training.Model, lightgbm.basic.Booster],
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
            self.weights = self.check_weights(self.weights)

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

    def uniq_feature_vals(self, df: pd.core.frame.DataFrame) -> List:
        '''
        Generates list of lists of all unique values across all protected class features of Pandas DataFrame.
        :return: list of lists unique values.
        '''
        combos = []
        for feature in self.config['protected_class_features']:
            combos.append(list(df[feature].unique()))
        return combos

    def sum_dict_vals(self, dictionary: Dict) -> float:
        '''
        Calculates the sum of the values of a non-nested dictionary.
        :param dictionary: a non-nested dictionary
        :return: float sum total
        '''
        sum_tot = 0
        for _, val in dictionary.items():
            _ = _ ; sum_tot += val
        return sum_tot

    def joint_distrib_dict(self, marginal_dict: Dict, mapping: Dict) -> Dict:
        '''
        Generates a dictionary of joint distributions from a supplied dictionary of marginal distributions assuming independence.
            If sum of joint distribution is not equal to 1.0, performs adjustment to ensure this is so.
        :param marginal_dict: dictionary of feature marginals
        :param mapping: dictionary of mappings between column names (numbers) and the combinations of protected class
            feature values from self.iterate_predictions()
        :return: dictionary of joint distributions where keys pertain to column names in dataframe from self.iterate_predictions()
        '''
        joint_dict = {}
        for key in mapping.keys():
            joint_dict[key] = 1
            for x in range(len(mapping[key])):
                combo = mapping[key][x] # individual feature-value combination
                joint_dict[key] *= marginal_dict[combo[0]][float(''.join(combo[1]))] # if key not in data will throw error

        # Ensure joint distribution sums to 1
        sum_tot = self.sum_dict_vals(joint_dict)

        if sum_tot != 1.0:
            for key in joint_dict.keys():
                joint_dict[key] = joint_dict[key] / sum_tot # rescale to enfore joint distributions sum to 1

        return joint_dict

    def iterate_predictions(self) -> Tuple[pd.core.frame.DataFrame, Dict]:
        '''
        Method iteratively generates an N x 1 vector of counterfactual predictions for all combinations of values for
            all protected class features. On each iteration, the protected class feature(s) for all observations are
            assigned the value(s) of that combination and a prediction is made. Each pd.Series prediction is concatenated
            onto a dataframe with the column name (e.g. 0, 1, ...) denoting a particular combination.
        :return: Pandas dataframe of counterfactual predictions and a dictionary mapping of columns to combinations
        '''

        df = self.ensure_dataframe(self.df) # convert to pd.DataFrame if not already
        combinations = list(product(*self.uniq_feature_vals(df))) # get list of lists of all protected class categorical values

        predictions = pd.DataFrame()
        temp = copy.deepcopy(df)
        for i in range(len(combinations)): # iterate across list of tuples, where each tuple is unique combination of protected class feature values
            temp[self.config['protected_class_features']] = combinations[i]
            predictions = pd.concat([predictions, pd.DataFrame(self.model.predict(temp), index=df.index).rename(
                columns={0: i})], axis=1)

        # Create mapping of column numbers to particular feature-value combinations
        # Returns dictionary of tuples for combo (each set of tuples within a list)
        mapping = {}
        for i in predictions.columns:
            mapping[i] = []
            index = 0  # positional index to map protected class feature name in config to value in combinations
            for feature in self.config['protected_class_features']:  # get protected class feature order
                mapping[i].append(tuple((feature, str(combinations[i][index]))))
                index += 1

        return (predictions, mapping)

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

    def weighted_predictions(self, joint_dict: Dict, prediction_df: pd.core.frame.DataFrame) -> pd.core.series.Series:
        '''
        Obtains weighted predictions, with weights corresponding to the joint distributions of the training set or custom weights.
        :param joint_dict: dictionary of joint distributions corresponding to columns in `prediction_df`
        :param prediction_df: counterfactual predictions from self.iterate_predictions() method
        :return: Pandas Series of predictions, weighted to reflect joint distributions of train or df
        '''
        wt_pred = np.zeros(shape=len(prediction_df))
        for feature, val in joint_dict.items():
            wt_pred += prediction_df[feature] * val
        return wt_pred

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

    def predictions(self) -> pd.core.frame.DataFrame:
        '''
        Generates predictions by calling methods of class.
        :return: Pandas dataframe of 2-4 columns of predictions:
            'unadj_pred' - unadjusted predictions for self.df
            'unif_wts' - predictions with uniform weights (i.e. simple average across N x V matrix of predictions)
                optionally:
                'pop_wts' - predictions weighted to reflect the marginal distribution in the training set (if provided)
                'cust_wts' - predictions with user-specified marginal weights.
        '''

        # Get counterfactual predictions
        iterated_predictions, mapping = self.iterate_predictions()

        # Marginals for either train or df
        if self.train is not None:
            marginals = self.feature_marginals(self.ensure_dataframe(self.train))  # marginals from training set to reweight test with same composition
        else:
            marginals = self.feature_marginals(self.ensure_dataframe(self.df))

        # Output dataframe of adjusted final predictions
        output_predictions = pd.DataFrame()
        output_predictions['unadj_pred'] = self.unadjusted_prediction()
        output_predictions['unif_wts'] = iterated_predictions.mean(axis=1)  # uniform weights (i.e. simple average)

        # Generate population weights from self.train if supplied
        if self.train is not None:
            # Get joint distributions for iterated_predictions dataframe
            joint_dict = self.joint_distrib_dict(marginals, mapping)
            output_predictions['pop_wts'] = self.weighted_predictions(joint_dict, iterated_predictions)  # weighted to reflect joint distributions of train or df

        # Dictionary of custom weights that combine user-supplied weights with marginals of either train or df
        if self.weights is not None:

            self.check_for_onehot()  # check if one-hot vectors possibly present and warn

            # Append missing protected class feature marginals, if any missing
            for key in [i for i in marginals.keys() if i not in self.weights.keys()]:
                self.weights.update({key: marginals[key]})

            # Get joint distributions using custom weights
            joint_dict_cust = self.joint_distrib_dict(self.weights, mapping)

            # Produce final N x 1 vector of custom weighted predictions
            output_predictions['cust_wts'] = self.weighted_predictions(joint_dict_cust, iterated_predictions)

        return output_predictions