import pandas as pd
import numpy as np
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