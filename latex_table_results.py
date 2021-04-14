import pandas as pd
import numpy as np
import lightgbm as lgb
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, precision_recall_curve
from DiscriminationMitigation import DiscriminationMitigator
import tabulate

pd.set_option('display.max_columns', 50)

def binary_metrics(y_true, y_pred):
    '''
    Calculates binary classification performance metrics for a given model.
    :param y_true: array_like, truth values as int
    :param y_pred: array_like, predicted values as int
    :returns: dict, with keys for each metric:
        accuracy - proportion of correct predictions out of total predictions
        sensitivity - (aka recall), of all true positives reviews how many did we correctly predict as positive
        specificity - (aka selectivity/TNR), of all true negatives how many did we correctly predict as negative
        precision - of all predicted positive cases how many were actually positive
        F-1 score - harmonic/weighted mean of precision and sensitivity scores
        ROC-AUC - area under receiver operating characteristic curve

    '''
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {}
    metrics['accuracy'] = round((tp + tn) / len(y_true), 4)
    metrics['sensitivity/recall'] = round(tp / (fn + tp), 4)  # aka recall
    metrics['specificity'] = round(tn / (tn + fp), 4)  # aka TNR
    metrics['precision'] = round(tp / (tp + fp), 4)
    metrics['f1'] = round(2 * (metrics['precision'] * metrics['sensitivity/recall']) \
                          / (metrics['precision'] + metrics['sensitivity/recall']), 4)
    metrics['roc_auc'] = round(roc_auc_score(y_true, y_pred), 4)

    return metrics

def continuous_metrics(y_true, y_pred):
    '''
    Calculates performance metrics for a continuous outcome from a model.
    :param y_true: array_like, truth values as float
    :param y_pred: array_like, predicted values as float
    :returns: dict, with keys for each metric:
        mae - Mean Absolute Error
        mse - Mean Squared Error
        rmse - Root Mean Squared Error
        r2 - Coefficient of Determination (r-squared)
    '''
    metrics = {}
    metrics['mae'] = round(sum(abs(y_true - y_pred)) / len(y_true), 4)
    metrics['mse'] = round(sum((y_true - y_pred) ** 2) / len(y_true), 4)
    metrics['rmse'] = round(np.sqrt(metrics['mse']), 4)
    metrics['r2'] = round(1 - (sum((y_true - y_pred) ** 2) / sum((y_true.mean() - y_true) ** 2)), 4)

    return metrics

def combine_prediction(X_test, y_test, pred, outcome='50k'):
    '''
    Combines X_test dataframe with np.array of predictions
    :param X_test: pd.DataFrame, X_test dataset
    :param pred: np.array, predicted y values
    :param outcome: str, column name to rename true y values
    :returns:
        pd.DataFrame
    '''
    X_test_combined = pd.concat([X_test, pd.DataFrame(y_test, columns=[outcome])], axis=1)
    return pd.concat([X_test_combined.reset_index(drop=True), pd.DataFrame(pred, columns=['pred'])], axis=1)

def read_process_data():

    ##########################
    #####   read data    #####
    ##########################
    df = pd.read_csv('./data/asec_2019.csv')

    ##########################
    ##### restrict data ######
    ##########################

    # Drop individuals with no employment or earnings last year
    df = df.loc[df['INCWAGE'] > 0]  # No wage/salary last year
    df = df.loc[(df['WKSWORK1'] > 0) & (df['WKSWORK1'] <= 52)]  # No weeks worked last year
    df = df.loc[(df['UHRSWORKLY'] > 0) & (df['UHRSWORKLY'] < 999)]  # No usual hours/week worked last year

    # Restrict to individuals aged 18-64
    df = df.loc[(df['AGE'] >= 18) & (df['AGE'] <= 64)]

    # Restrict to non-Hispanics
    df = df.loc[df['HISPAN'] == 0]

    # Restrict to non-mixed-race Blacks and Whites
    df = df.loc[df.RACE.isin([100, 200])]
    df['blk'] = np.where(df['RACE'] == 200, 1, 0)

    # Restrict to adult civilians
    df = df.loc[df['POPSTAT'] == 1]

    ##########################
    ### engineer features  ###
    ##########################

    # Flag for part-time usual work
    df['pt'] = np.where(df['UHRSWORKLY'] < 35, 1, 0)

    # Non-linearities for age
    df['AGE2'] = df['AGE'] ** 2
    df['AGE3'] = df['AGE'] ** 3

    # Ensure education coded correctly
    df = df.loc[df['EDUC'] <= 125]  # 999 is missing

    # Hourly wage
    df['hrwage'] = df['INCWAGE'] / (df['WKSWORK1'] * df['UHRSWORKLY'])

    # Restrict to people earning 1 < hr_wage < 100
    df = df.loc[(df['hrwage'] > 1) & (df['hrwage'] < 100)]

    # Log hourly wage
    df['lnwage'] = np.log(df['hrwage'])

    # Flag for whether total earnings > 50,000 or not
    df['50k'] = np.where(df['INCWAGE'] > 50000, 1, 0)

    df = df[['lnwage', 'hrwage', '50k', 'pt', 'INCWAGE', 'WKSWORK1', 'UHRSWORKLY',
             'AGE', 'AGE2', 'AGE3', 'SEX', 'blk', 'MARST', 'SCHLCOLL', 'EDUC']]

    # convert categorical features to pandas category dtype
    categorical_features = ['SEX', 'MARST', 'SCHLCOLL', 'EDUC', 'pt', 'blk']
    for col in categorical_features:
        df[col] = df[col].astype('category')

    return df

def reg_table_descriptives():

    count = 1
    results = pd.DataFrame()
    for table in [table1, table2, table3, table4]:

        # Restrict to Black dummy and intercept, also reverses sort index
        table[1] = table[1].iloc[:2, ].sort_index()
        table[1] = table[1].rename(index={'blk': 'Black'})

        coef = table[1].iloc[:, 0].map('{:.3f}'.format)  # round to three decimal places or add trailing zeros as needed
        se = table[1].iloc[:, 1].map('{:.3f}'.format)

        # significance stars for 95% 99%, 99.9% confidence, 2-tailed t-test
        for i in range(len(coef)):
            if table[1].iloc[i, 3] < 0.0005:
                coef.iloc[i] = coef.iloc[i] + '***'
            elif table[1].iloc[i, 3] < 0.005:
                coef.iloc[i] = coef.iloc[i] + '** '
            elif table[1].iloc[i, 3] < 0.025:
                coef.iloc[i] = coef.iloc[i] + '*  '
            else:
                coef.iloc[i] = coef.iloc[i] + '   '

        # create 1d dataframe of stacked coefficients and standard errors
        name = '(' + str(count) + ')'
        temp = pd.DataFrame(columns=[name])
        for row in range(len(table[1])):
            temp = pd.concat([temp, pd.DataFrame({name: coef.iloc[row]}, index=[coef.index[row]])])
            temp = pd.concat([temp, pd.DataFrame({name: '('+se.iloc[row]+')'}, index=[''])])

        # whether controls present or not
        controls_dict = {1: 'False',
                         2: 'True',
                         3: 'False',
                         4: 'True'}
        controls = pd.DataFrame({name: controls_dict[count]}, index=['Includes controls'])
        temp = pd.concat([temp, controls], axis=0)

        # model diagnostics
        model_type = pd.DataFrame({name: table[0].iloc[0, 1]}, index=['Estimator'])
        dep_var = pd.DataFrame({name: table[0].iloc[1, 1]}, index=['Dependent Variable'])
        observations = pd.DataFrame({name: '{:,}'.format(int(table[0].iloc[3, 1]))}, index=['N'])
        r2 = pd.DataFrame({name: table[0].iloc[6, 1]}, index=['R-squared'])
        for col in [model_type, dep_var, observations, r2]:
            temp = pd.concat([temp, col], axis=0, ignore_index=False)

        # Attach finally to combined results
        results = pd.concat([results, temp], axis=1)

        count += 1

    return results

if __name__ == '__main__':

    df = read_process_data()

    ######################################
    #####   Discriptive Differences  #####
    ######################################

    # Table 1
    y = df['50k'].reset_index(drop=True)
    predictors = df['blk'].astype('int').reset_index(drop=True)
    predictors = sm.add_constant(predictors)  # add constant
    table1 = sm.OLS(y, predictors, hasconst=True).fit(cov_type='HC3', use_t=True).summary2().tables

    # Table 2
    y = df['50k'].reset_index(drop=True)
    predictors = df[['blk', 'AGE', 'AGE2', 'AGE3', 'pt']].reset_index(drop=True)
    predictors['blk'] = predictors['blk'].astype(int)
    predictors['pt'] = predictors['pt'].astype(int)
    for col in ['SEX', 'MARST', 'SCHLCOLL', 'EDUC']:
        onehot_vector = pd.DataFrame(OneHotEncoder().fit_transform(df[[col]]).toarray()[:, 1:])
        onehot_vector.columns = [str(i) + '_' + col[:3] for i in onehot_vector.columns]
        predictors = pd.concat([predictors, onehot_vector], axis=1)
    predictors = sm.add_constant(predictors)  # add constant
    table2 = sm.OLS(y, predictors, hasconst=True).fit(cov_type='HC3', use_t=True).summary2().tables

    # Table 3
    y = df['lnwage'].reset_index(drop=True)
    predictors = df['blk'].astype('int').reset_index(drop=True)
    predictors = sm.add_constant(predictors)  # add constant
    table3 = sm.OLS(y, predictors, hasconst=True).fit(cov_type='HC3', use_t=True).summary2().tables

    # Table 4
    y = df['lnwage'].reset_index(drop=True)
    predictors = df[['blk', 'AGE', 'AGE2', 'AGE3', 'pt']].reset_index(drop=True)
    predictors['blk'] = predictors['blk'].astype(int)
    predictors['pt'] = predictors['pt'].astype(int)
    for col in ['SEX', 'MARST', 'SCHLCOLL', 'EDUC']:
        onehot_vector = pd.DataFrame(OneHotEncoder().fit_transform(df[[col]]).toarray()[:, 1:])
        onehot_vector.columns = [str(i) + '_' + col[:3] for i in onehot_vector.columns]
        predictors = pd.concat([predictors, onehot_vector], axis=1)
    predictors = sm.add_constant(predictors)  # add constant
    table4 = sm.OLS(y, predictors, hasconst=True).fit(cov_type='HC3', use_t=True).summary2().tables

    results = reg_table_descriptives()
    print(results.to_latex()) # edited further by hand



