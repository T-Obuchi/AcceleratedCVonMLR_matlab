# AcceleratedCVonMLR_matlab
Approximate cross-validation for multinomial logistic regression with elastic net regularization

This is free software, you can redistribute it and/or modify it under the terms of the GNU General Public License, version 3 or above. See LICENSE.txt for details.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

# DESCRIPTION
Using estimated weight vectors *wV* given the feature data *X* and the class *Ycode* for multinomial logistic regression penalized by elastic net regularization (*L1* norm and *L2* norm), this program computes and returns an approximate leave-one-out estimator (LOOE) and its standard error of predictive likelihood. All required codes are in the "routine" folder. Note that this program itself does not contain any solver to obtain *wV*. Please use other distributed programs for the purpose.

# USAGE
For multinomial logistic regression with *Np* (*>2*) classes,
```matlab
[LOOE,ERR] = acv_mlr(wV,X,Ycode,Np,lambda2)
```
Inputs:
- *wV*: the estimated weight vectors of *N* * *Np* dimensional
- *X*: the set of feature vectors  of *M* * *N* dimensional
- *Ycode*: the *M* * *Np* dimensional binary matrix representing the class to which the corresponding feature vector belongs
- *lambda2*: the coefficient of the *L2* norm. If this argument is absent, *lambda2* is set to be the default value *lambda2=0*

Outputs:
- *LOOE*: Approximate value of the leave-one-out estimator
- *ERR*: Approximate standard error of the leave-one-out estimator

For more details, type help acv_mlr.

If *acv_mlr* takes a long running time, please try a further simplified approximation as
```matlab
[LOOE,ERR] = saacv_mlr(wV,X,Ycode,Np,lambda2)
```
In our experiments, *acv_mlr* runs faster if *N* is several hundreds or less, but *saacv_mlr* is faster for larger *N*. For small data and model, these approximations are not necessarily fast, and hence we recommend to perform the literal cross-validation for such cases. For details, see REFERENCE.

For binomial logistic regression (logit model),
```matlab
[LOOE,ERR] = acv_logit(w,X,Ycode,lambda2)
```
Inputs:
- *w*: the estimated weight vector of *N* dimensional
- *X*: the set of feature vectors  of *M* * *N* dimensional
- *Ycode*: the *M* * *2* dimensional binary matrix representing the class to which the corresponding feature vector belongs
- *lambda2*: the coefficient of the *L2* norm. If this argument is absent, *lambda2* is set to be the default value *lambda2=0*

Outputs are the same as the multinomial case. The  further simplified approximation for the logit model is implemented in *saacv_logit*.

# DEMONSTRATION
In the "demo" folder, demonstration codes for the multinomial and binomial logistic regressions, demo_LOOEapprox_mlr.m and demo_LOOEapprox_logit.m, respectively, are available.

**Requirement**: glmnet (https://web.stanford.edu/~hastie/glmnet_matlab/) is required to obtain the weight vectors in these demonstration codes.

# REFERENCE
Tomoyuki Obuchi and Yoshiyuki Kabashima: "Accelerating Cross-Validation in Multinomial Logistic Regression with $ell_1$-Regularization", arXiv: 1711.05420
