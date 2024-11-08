import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.formula.api import glm, ols
from statsmodels.api import families
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.discrete.discrete_model import Poisson, NegativeBinomial, Logit
from statsmodels.genmod.families import Poisson as sm_Poisson, Binomial
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools import add_constant
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression

# Logistic regression and odds calculations
# -----------------
# GLM binary logistic regression for multiple regression (log odds as outputs)
model_x = glm('binary_variable_x ~ continuous_variable + var_x + var_y', 
              data=df_x, family=Binomial()).fit()

# Predicted odds
odds = np.exp(model_x.params)

# Predicted probability
prob = model_x.predict(df_x)

# Odds to probability conversion
def odds_to_prob(odds):
    return odds / (1 + odds)

# Probability to odds conversion
def prob_to_odds(p):
    return p / (1 - p)

# Log odds (logit) to probability
def logit_to_prob(logit):
    return 1 / (1 + np.exp(-logit))

# Probability to log odds
def prob_to_logit(p):
    return np.log(p / (1 - p))

# Binary dataset creation
# -----------------
df_x['new_variable'] = df_x['variable_x'] > x

# Creating new variables with conditions
# -----------------
df_x['new_var'] = np.where(df_x['variable'] == 1, 'x', 'y')
doctor_df['gender'] = np.where(df_DoctorAUS['sex'] == 1, 'female', 'male')

# Multilevel binary logistic regression
# -----------------
from statsmodels.regression.mixed_linear_model import MixedLM
model_x = MixedLM.from_formula('binary_variable_x ~ continuous_variable + var_x + var_y', 
                               groups=df_x['subject'], 
                               data=df_x, 
                               family=Binomial()).fit()

# Predictions and calculations for model parameters
# -----------------
# Coefficients as log odds
b = model_x.params
# To calculate log odds, manually plug in coefficients with variables
log_odds = b['Intercept'] + b['var_x'] * x + b['var_y'] * x

# Poisson Regression
# -----------------
# Poisson regression to estimate rates
poisson_model = glm('x ~ y + z', data=df_x, family=sm_Poisson()).fit()
# If exposure variable needed, use an offset
poisson_exposure_model = glm('x ~ y + offset(np.log(z))', data=df_x, family=sm_Poisson()).fit()

# Negative Binomial Regression
# -----------------
# Handling overdispersed count data with Negative Binomial regression
neg_binom_model = NegativeBinomial.from_formula('x ~ y + ...', data=df_x).fit()

# Predictions with Zero-Inflated Models
# -----------------
# Zero-inflated Poisson regression
from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP

# Zero-inflated Poisson
zip_model = ZeroInflatedPoisson.from_formula('x ~ y + ...', data=df_x).fit()

# Zero-inflated Negative Binomial
zinb_model = ZeroInflatedNegativeBinomialP.from_formula('x ~ y + ...', data=df_x).fit()

# Predictions for count and zero parts
# Zero (probability of zero outcomes)
zero_pred = zip_model.predict(df_x, which='zero')
# Count (expected count)
count_pred = zip_model.predict(df_x, which='count')

# Model Deviance and Comparison
# -----------------
# Residual deviance
deviance = -2 * zip_model.llf
# Compare models using likelihood ratio tests
anova_results = zip_model.compare_lr_test(other_model)

# Calculating Confidence Intervals for Odds Ratios
# -----------------
conf_int = np.exp(model_x.conf_int())

# Risk Ratios and Odds Ratios
# -----------------
def odds_ratio(p_x, p_y):
    return (p_x / (1 - p_x)) / (p_y / (1 - p_y))

def risk_ratio(p_x, p_y):
    return p_x / (p_x + p_y)

# Poisson Maximum Likelihood
# -----------------
# Poisson probability mass function
def poisson_pmf(k, lambda_):
    return (np.exp(-lambda_) * lambda_**k) / np.math.factorial(k)

# Zero-inflated Poisson model probability function
def zero_inflated_pmf(x, lambda_, zero_prob):
    if x == 0:
        return zero_prob + (1 - zero_prob) * poisson_pmf(0, lambda_)
    else:
        return (1 - zero_prob) * poisson_pmf(x, lambda_)

# Deviance from likelihood
# -----------------
def deviance_from_likelihood(log_likelihood):
    return -2 * log_likelihood

# Example prediction using a fitted model
# -----------------
# Create new data for prediction
new_data = pd.DataFrame({'cont_var': [x], 'categorical_var': ['x']})
new_data['predicted_log_odds'] = model_x.predict(new_data)
new_data['predicted_prob'] = model_x.predict(new_data, type='response')

# Zero-inflated model predictions
# -----------------
# Count predictions for zero-inflated models
count_pred = zinb_model.predict(df_x, which='count')
zero_pred = zinb_model.predict(df_x, which='zero')

# Predicted probability of zero-inflated poisson count
def zero_infl_pred(model, intercept, coef, x):
    return np.exp(intercept + coef * x) * (1 - logit_to_prob(intercept + coef * x))

