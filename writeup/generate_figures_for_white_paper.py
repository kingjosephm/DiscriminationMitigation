
import json
from DiscriminationMitigation import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


with open('example_config.json') as j:
    config = json.load(j)

# dgp parameterization
N = 8000
class_prob=0.5
gamma=np.matrix([[2,3.5],[.4,.6]])
alpha=np.matrix([[0,2]])
beta=np.matrix([[2,2],[1,1]])

############################
# define data generation function, interpretive function, model functions
def _dgp(n=500, class_prob=0.5,
                 gamma=np.matrix([[2, 3], [.4, .6]]),
                 alpha=np.matrix([[0, 2]]),
                 beta=np.matrix([[2, 2], [1, 1]])):
    np.random.seed(1776)

    # Protected class variable -- dichotmous.  Two variables are mutually exclusive.
    c1 = np.random.binomial(1, p=class_prob, size=n)  # group 1
    c0 = 1 - c1  # group 0

    # Other covariates
    w0 = gamma[0, 0] * c0 + gamma[0, 1] * c1 + np.random.normal(0, 0.5, size=n)  # linear function of class & shock
    w1 = gamma[1, 0] * c0 + gamma[1, 1] * c1 + np.random.normal(0, 0.3, size=n)

    # Outcome variable
    y = alpha[0, 0] * c0 + alpha[0, 1] * c1 + \
        beta[0, 0] * c0 * w0 + beta[0, 1] * c1 * w0 + \
        beta[1, 0] * c0 * w1 + beta[1, 1] * c1 * w1 + \
        np.random.normal(0, 0.3, size=n)

    return pd.DataFrame([y, c0, c1, w0, w1]).T.rename(columns={0: 'y', 1: 'c0', 2: 'c1', 3: 'w0', 4: 'w1'})

# interpretive function.  helps to generate lines showing what the model is doing
def _observable_y(c0, c1, w0, w1, alpha, beta):
    """
    This function 'predicts' y as a funciton of the observables
    and parameters.  That is, it is what y would be if the shocks were zero.
    This function is used to generate lines for the plots.
      """
    y = alpha[0,0]*c0   + alpha[0,1]*c1 + \
        beta[0,0]*c0*w0 + beta[0,1]*c1*w0 + \
        beta[1,0]*c0*w1 + beta[1,1]*c1*w1
    return y




# ml model with and without class

tf.keras.backend.clear_session()
model_w_class = tf.keras.Sequential()
model_w_class.add(tf.keras.layers.Input(shape=4,))
model_w_class.add(tf.keras.layers.Dense(32))
model_w_class.add(tf.keras.layers.Dropout(0.05))
model_w_class.add(tf.keras.layers.Dense(16))
model_w_class.add(tf.keras.layers.Dropout(0.05))
model_w_class.add(tf.keras.layers.Dense(16))
model_w_class.add(tf.keras.layers.Dense(1))

model_w_class.compile(optimizer='adam', loss='mse')

tf.keras.backend.clear_session()
model_wo_class = tf.keras.Sequential()
model_wo_class.add(tf.keras.layers.Input(shape=2,))
model_wo_class.add(tf.keras.layers.Dense(32))
model_wo_class.add(tf.keras.layers.Dropout(0.05))
model_wo_class.add(tf.keras.layers.Dense(16))
model_wo_class.add(tf.keras.layers.Dropout(0.05))
model_wo_class.add(tf.keras.layers.Dense(16))
model_wo_class.add(tf.keras.layers.Dense(1))

model_wo_class.compile(optimizer='adam', loss='mse')

##################################################
# generate the data, split for models, train models
data = _dgp(n=N, class_prob=class_prob, gamma=gamma, alpha=alpha, beta=beta)

X_train, X_test, y_train, y_test = train_test_split(data.loc[:, ~data.columns.isin(config['target_feature'])],
                                                    data[config['target_feature']], random_state=1776,
                                                    test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=123, test_size=0.2)
model_w_class.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val))


X_train_wo_class, X_test_wo_class, y_train, y_test = train_test_split(data.loc[:, ~data.columns.isin(config['target_feature'] + ["c0", "c1"])],
                                                    data[config['target_feature']], random_state=1776,
                                                    test_size=0.2)
X_train_wo_class, X_val_wo_class, y_train, y_val = train_test_split(X_train_wo_class, y_train, random_state=123, test_size=0.2)
model_wo_class.fit(X_train_wo_class, y_train, epochs=10, batch_size=64, validation_data=(X_val_wo_class, y_val))

#######################################################################################
# create the figures

# fig1
synth_vars = ["y", "c0", "c1", "w0", "w1"]
c0_data = data.loc[data.loc[:, "c0"] == 1, :]
c1_data = data.loc[data.loc[:, "c1"] == 1, :]

#matplotlib.use('module://ipykernel.pylab.backend_inline')
fig1, axs = plt.subplots(ncols=1)
axs.scatter(c0_data.loc[:, "w0"],c0_data.loc[:, "y"],color='blue', marker= '.', label='Group 0', s=0.01)
axs.scatter(c1_data.loc[:, "w0"],c1_data.loc[:, "y"],color= 'red', marker='.', label='Group 1', s=0.01)

axs.set_title('Different intercepts for different groups')
axs.set_xlabel('w0')
axs.set_ylabel('y')
axs.legend(loc='best')
#####################################################################################
# fig 2 -- on fig1, overlay lines and predictions for model with class, no mitigation
# begin base scatterplot ########
synth_vars = ["y", "c0", "c1", "w0", "w1"]
c0_data = data.loc[data.loc[:, "c0"] == 1, :]
c1_data = data.loc[data.loc[:, "c1"] == 1, :]

#matplotlib.use('module://ipykernel.pylab.backend_inline')
fig2, axs = plt.subplots(ncols=1)
axs.scatter(c0_data.loc[:, "w0"],c0_data.loc[:, "y"],color='blue', marker= '.', label='Group 0', s=0.01)
axs.scatter(c1_data.loc[:, "w0"],c1_data.loc[:, "y"],color= 'red', marker='.', label='Group 1', s=0.01)

axs.set_title('Different intercepts for different groups')
axs.set_xlabel('w0')
axs.set_ylabel('y')
axs.legend(loc='best')
# end base scatterplot #####

def y_fit_of_w0_of_c1(w0, c1):
    mean_w1 = (1-c1)*gamma[1,0] + c1*gamma[1,1]
    return _observable_y(1-c1, c1, w0, mean_w1, alpha, beta)

w0_space = np.linspace(0, 5, 20)
f_w0_c0 = lambda w0: y_fit_of_w0_of_c1(w0, 0)
f_w0_c1 = lambda w0: y_fit_of_w0_of_c1(w0, 1)
y_fitted_w0_c0 = f_w0_c0(w0_space)
y_fitted_w0_c1 = f_w0_c1(w0_space)

axs.plot(w0_space, y_fitted_w0_c0, color = "blue")
axs.plot(w0_space, y_fitted_w0_c1, color = "red")

y_hat = model_w_class.predict(X_test)
axs.scatter(X_test.loc[:,"w0"], y_test, color = "purple", marker= 'x',
            label='prediction: class omited', s=.2)



# fig 3 -- on fig1, overlay lines and predictions for model without class

# base scatterplot ###
fig3, axs = plt.subplots(ncols=1)
axs.scatter(c0_data.loc[:, "w0"],c0_data.loc[:, "y"],color='blue', marker= '.', label='Group 0', s=0.01)
axs.scatter(c1_data.loc[:, "w0"],c1_data.loc[:, "y"],color= 'red', marker='.', label='Group 1', s=0.01)

axs.set_title('Different intercepts for different groups')
axs.set_xlabel('w0')
axs.set_ylabel('y')
axs.legend(loc='best')
# end base scatterplot ###

y_hat_wo_class = model_wo_class.predict(X_test.loc[:,["w0", "w1"]])
axs.scatter(X_test.loc[:,["w0"]], y_hat_wo_class,color = "purple", marker= 'x',
            label='prediction: class omitted', s=.2)

y_hat = model_w_class.predict(X_test)
axs.scatter(X_test.loc[:,"w0"], y_test, color = "purple", marker= 'x',
            label='prediction: class omited', s=.2)

w1_mean = (gamma[1,0] + gamma[1,1])/2
X_hypotheticals__to_draw_line = pd.DataFrame({"w0": w0_space, "w1": w0_space * 0 + w1_mean})
y_hypothetical = model_wo_class.predict(X_hypotheticals__to_draw_line)
axs.plot(X_hypotheticals__to_draw_line.loc[:,"w0"], y_hypothetical, color="orange")

#w1_mean = (gamma[1,0] + gamma[1,1])/2
#X_hypotheticals__to_draw_line = pd.DataFrame({"w0": w0_space, "w1": w0_space * 0 + w1_mean})

# fig 4 -- on fig1, overlay lines and predictions for model with class, with mitigation
# base scatterplot ###
fig3, axs = plt.subplots(ncols=1)
axs.scatter(c0_data.loc[:, "w0"],c0_data.loc[:, "y"],color='blue', marker= '.', label='Group 0', s=0.01)
axs.scatter(c1_data.loc[:, "w0"],c1_data.loc[:, "y"],color= 'red', marker='.', label='Group 1', s=0.01)

axs.set_title('Different intercepts for different groups')
axs.set_xlabel('w0')
axs.set_ylabel('y')
axs.legend(loc='best')
# end base scatterplot ###

config = {"protected_class_features": ["c0", "c1"], "target_feature": "y"}
ex1 = DiscriminationMitigator(df=X_test, model=model_w_class, config=config, train=None, weights=None).predictions()

y_hat = model_w_class.predict(X_test)
axs.scatter(X_test.loc[:,"w0"], y_test, color = "purple", marker= 'x',
            label='prediction: class omited', s=.2)
