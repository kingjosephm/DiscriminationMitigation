## Discrimination Mitigator

Motivated by the Frisch-Waugh-Lovell theorem, `DiscriminationMitigator` offers a simple, intuitive
method to mitigate potential discrimination associated with protected class features
(e.g. race, gender) in supervised machine learning algorithms. In general, it is not
possible to ensure a model does not discriminate on protected class(es) without making
strong assumptions about the data generating process. Instead, `DiscriminationMitigator`
takes a pre-trained ML model - **that included the protected class attribute(s)** - and
applies various weights to a series of counterfactual predictions per observation,
yielding adjusted predictions. Importantly, this method does not require advanced
statistical or programming knowledge, and can be used with two dominant Python ML
libraries: Tensorflow Keras and LightGBM. For example code, see `Example.html`.


#### Algorithm
Given the target variable *y*, a vector of protected class attributes *C* (*c∈C*) for
individual *i* (i = 1...*N*):
- For each unique value (*v∈V*) of each protected class, *c*:
    - Set c<sub>*i*</sub> = c<sub>*v*</sub> for all *N*
    - Generate an *N* x 1 vector of counterfactual predictions, ŷ<sub>*cv*</sub>
    - Save ŷ<sub>*cv*</sub> to dataframe Pred<sub>*counterfactual*</sub>
    - Repeat until end
- Pred<sub>*counterfactual*</sub> will be *N* x C<sub>*V*</sub> dimensions
- Apply weights to Pred<sub>*counterfactual*</sub>, yielding ŷ<sub>*adjusted*</sub>


#### Inputs
- `df` - Pandas DataFrame or list of Pandas Series/DataFrame(s) to generate
    adjusted predictions. Typically the test set.
- `model` - Pre-trained Tensorflow Keras Model or Sequential class models. ***Model
   must have been trained using all protected class feature(s)***
- `config` - JSON dictionary, see `Example.ipynb`. This **must** include keys for
  'protected_class_features' (i.e. a list of all protected class feature names)
  and 'target_feature'
- `train` (optional) - Pandas DataFrame or list of Pandas Series/DataFrame(s) used
    to weight the predictions to reflect the marginal distributions of *C*. This should be
    the same training set used to train `model`, but need not be. Importantly, if this
    dataframe includes a value for a protected class features that is not found in `df`
    (e.g. sparsely populated values or a small test set) this will trigger a UserWarning 
    and overlapping protected class values between `df` and `train` will be adjusted so 
    that the marginals for that feature sums to 1.    
- `weights` (optional) - JSON dictionary of user-specified marginal distributions, see
    `Example.ipynb`. Users can specify the custom marginals of anywhere between one and the full
    number of protected class features listed in `config`. Marginals from either `df` 
    or `train` (if supplied) are used for the remaining protected class features. *Note, 
    feature marginals must sum to 1*. The program will raise a Warning
    if two or more features in `df` appear to be one-hot vectors, reminding the user that the
    marginals of adjacent one-hot vectors should reflect mutual exclusivity. For example, if
    a binary variable is transformed into two one-hot vectors, the marginal distributions
    should be mirror opposites. The program, however, *does not* enforce this; thus users
    must ensure the custom marginals make sense.

#### Outputs
A Pandas Dataframe with 3 (or optionally 4) columns:
- `unadj_pred` - unadjusted predictions, potentially containing discriminatory effect
- `unif_wts` - uniform weights, i.e. a simple average across all counterfactual predictions
- `pop_wts` - population weights, i.e. weighted to reflect marginal distribution
  per feature in `df`. If `train` is included, this dataset will be used to
  calculate marginal distributions per feature
- `cust_wts` (optional) - if user supplies `weights`, this will report the predictions
  adjusted according to the specified marginal distributions per feature-value combination.

#### Requirements
See `requirements.txt`