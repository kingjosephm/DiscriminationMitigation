## Discrimination Mitigator

Motivated by the Frisch-Waugh-Lovell theorem, `DiscriminationMitigator` offers a simple, intuitive
method to mitigate potential discrimination associated with protected class features
(e.g. race, gender) in supervised machine learning algorithms. In general, it is not
possible to ensure a model does not, in some sense, discriminate on protected class(es) without making
strong assumptions about the data generating process.  
Crucially, omitting the protected class variables from the model will
not necessarily prevent the model from discriminating on the basis
of protected class membership.
This is because other variables might proxy for protected class membership;
these variables might predict a target partly through the protected class
for which they proxy.    

`DiscriminationMitigator` ameliorates proxy discrimination by
taking a pre-trained ML model that was trained using the protected class attribute(s)
and averages predictions across combinations of counterfactional protected class memberships.
Because protected class is included in the pre-trained model,
the extent to which other variables predict protected class membership does
not contribute to their prediction of the target. Averaging predictions
across protected class counterfactuals ensures the actual protected class membership does
not contribute to the forecast of the target.    


Importantly, this method does not require advanced
statistical or programming knowledge, and can be used with two dominant Python ML
libraries: Tensorflow Keras and LightGBM. For example code, see `Example.ipynb`.


#### Algorithm
Given the target variable *y*, a vector of protected class attributes *C* (*c∈C*), and
individual *i* (i = 1...*N*):
- For each combination of protected class feature values (*v∈V*):
    - Set C = v for all *N*
    - Generate an *N* x 1 vector of counterfactual predictions, ŷ<sub>*v*</sub>
    - Save ŷ<sub>*v*</sub> to dataframe Pred<sub>*counterfactual*</sub>
    - Repeat until end
- Pred<sub>*counterfactual*</sub> will be *N* x *V* dimensions
- Apply weights to Pred<sub>*counterfactual*</sub>, yielding ŷ<sub>*adjusted*</sub>


#### Inputs
- `df` - Pandas DataFrame to generate adjusted predictions. Typically the test set.
- `model` - Pre-trained LightGBM or Tensorflow Keras (Model/Sequential) model. ***Model
   must have been trained using all protected class feature(s)***
- `config` - JSON dictionary, see `Example.ipynb`. This **must** include keys for
  'protected_class_features' (i.e. a list of all protected class feature names).
- `train` (optional) - Pandas DataFrame used to weight predictions according
    to the joint distributions of protected class features in another dataset. This may be 
    advantageous, for example, if your test set is considerably smaller than your training set.  
- `weights` (optional) - JSON dictionary of user-specified marginal distributions, see
    `Example.ipynb`. Users can specify the custom marginals of anywhere between one and the full
    number of protected class features listed in `config`. Marginals from either `df`
    or `train` (if supplied) are used for the remaining protected class features, and assuming
    independence, the joint distributions are calculated from these. *Note,
    feature marginals must sum to 1*. The program will also raise a warning
    if two or more features in `df` appear to be one-hot vectors, reminding the user that the
    marginals of adjacent one-hot vectors should reflect mutual exclusivity. For example, if
    a binary variable is transformed into two one-hot vectors, their marginal distributions
    should be polar opposites. The program, however, *does not* enforce this; thus users
    must ensure the custom marginals make sense. At present it is only possible to reweight 
    the marginal distributions of select features, not the joint distributions of combinations
    of features.

#### Outputs
A Pandas Dataframe with between 2-4 columns:
- `unadj_pred` - unadjusted predictions, potentially containing discriminatory effect
- `unif_wts` - uniform weights, i.e. a simple average across all counterfactual predictions using the
    joint distributions of protected class features in `df`.
- `pop_wts` (optional) - if user supplies `train`, this will contain adjusted predictions weighted 
    to reflect joint distributions of protected class features in that dataset.
- `cust_wts` (optional) - if user supplies `weights`, this will report the predictions
  adjusted according to the specified marginal distributions per feature-value combination.

#### Requirements
See `requirements.txt`
