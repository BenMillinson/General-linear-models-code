library(tidyverse)
library(modelr) #for add_predictions
library(MASS)
library(lme4)
library(pscl)
library(lmerTest)
library(dplyr)

#log odds, odds, probability from models-----
#glm beta outputs are the predicted log odds
#for odds use exp(model$coefficients)
#for probability use plogis(model)

#creating a binary data set----
#(where x is a defined value to split the sample into 2 groups, by the defined variable)
new_df <- mutate(df_x, variable = variable_x > x)

#creating new variables----
new_df <- mutate(df_x, variable = ifelse(variable == 1, 'x', 'x'))
#e.g
doctor_df <- mutate(df_DoctorAUS, gender = ifelse(sex == 1, 'female', 'male'))
#where female = 1 and male = 0

#binary logistic regression (multiple regression)----
Model_x <- glm(binary_variable_x ~ continuous_variable
               + var_x + var_y, data = df_x, family = binomial(link = 'logit'))
#which gives the log odds

# Multilevel binary logistic regression----
Model_x <- glmer(binary_variable_x ~ continuous_variable + var_x + var_y + 
                   (1 | subject),  # Random intercept for each subject
                 data = df_x, 
                 family = binomial(link = 'logit'))

#odds from logistic regression----
exp(model_x)

#probability from logistic regression----
#same using the coefficents
plogis(model_x)

#odds from probability----
p/(1-p)

#probability from odds----
p=odds/(1+odds)

#probability from log odds----
1/(1+e^-logit)
logit <- x
1/(1+exp(-logit))
#or more simply,
plogis(x)

#log odds (logit) from probability----
log(p/(1-p))
#or
log(theta/(1-theta))

#odds from log odds----
log(theta/(1-theta)) #becomes:
theta/(1-theta) #through the function
e^logit #which is
exp(logit)

#odds ratio----
(p_x/(1-p_x))/(p_y/(1-p_y))

#from a model:
M_x <- glm(var_x ~ var_y + var_z...,
           data = df_x,
           family = binomial)
M_x
exp(M_x$coefficients)

#risk ratio----
P(p) = (p/p+q)

#log likelihood of a model----
logLik(M)

#loglikehood ratio----
log(x/y)

#prediction with model coefficients----
#where x is what you're predicting
b <- coef(model_x)
b['(Intercept)'] + b['var_x'] * x + b['var_y'] * x...

#making predictions for models----
#which gives predicted log odds
newdf_x <- tibble(cont_var = x, categorical_var = 'x')
add_predictions(newdf_x, model_x)

#getting predicted probabilities----

newdf_x <- tibble(cont_var = x, categorical_var = 'x')
add_predictions(newdf_x, model_x, type = 'response')

#model deviance----
#also known as residual deviance
logLik(M) * -2
#log lik = log p(data|model)
deviance(M)

#model comparison----
deviance(M_x)-deviance(M_y)
#or better:
anova(M_y, M_x, M_z..., test = 'Chisq')

#confidence intervals (BLR)----

confint.default(M_x)

#confidence intervals for odds ratio----
#use this one for normal outputs
exp(confint.default(M_3_1))

#non-linear changes of units----
#The value of the coefficient for x is 0.25.
#By what factor does the odds of y (being equal to 1)
#increase for every unit change in x? 
exp(0.25)

#deviance from logarithms----
-2 * x

#log (of the) likelihood from deviance----
-(x/2)

#deviance from likelihood----
log(x) *-2

#log likelihood ratio-----
log(p_x/p_y)

#poisson regression----
#it is usually referred as the rate parameter
#it gives the variance of the poisson random variable
#it gives the mean value of the poisson random variable
#it is usually denoted by the greek letter lambda

#for rates(mean) it is exp(beta coefficients (which are log odds))

#the outcome variable is distributed as a poisson distribution, 
#and its parameter is an exponential function of a linear function of the predictors

#the outcome variable is distributed as a poisson distribution, 
#and the log of its parameter is a linear function of the predictors

#creating new variables----
new_df <- mutate(df_x, new_variable = ifelse(variable == 1, 'x', 'x'))

#poisson regression----
x <- glm(x ~ y + z +...,
         data = df_x, family = poisson(link = 'logit'))

#poisson regression glm with exposure variables----
#where offset is the exposure
x <- glm(x ~ y + ... + offset(log(z)),
         data = df_x,
         family = poisson())

#confidence intervals for rates----
exp(confint.default(model_x, parm='new subset variable', level = x))
#or better
exp(confint(x))

#e.g. from making a 'male' variable from 'gender'
exp(confint.default(x, parm='gendermale'))

#exposure variables ----
x <- glm(x ~ y +...+ offset(log(z)),
         data = df_x,
         family = poisson(link='log'))

#poisson maxmimum likelihood function----
dpois(k, lambda = x)
(exp(-lambda) * lambda^k/factorial(x))
#where lamda is the parameter and k is the probability X takes a value of x

#standard deviation of lambda----
sqrt(lambda)


#negative binomial regression----

M_x <- glm.nb(x ~ y + ..., data = df_x)

#probability in negative binomial----
#x is what youre finding and z is the dispersion parameter
dnbinom(x = x, mu = y, size = z)

#predictions in negative binomial regression----
df_x <- tibble(x = c(x))

add_predictions(df_x, M_x) #log of average 
add_predictions(df_x, M_x, type = 'response') #average 

#confidence intervals (mean)----
exp(confint.default(M_x))

#null deviance in negbin regression----
#model comparison (chi-square)----
M_null <- glm.nb(x ~ 1, data = df_x)
anova(M_x, M_null, test = 'Chisq')
# x^2 is the LR stat

#logliklihood of null models----
logLik(M_null)

#deviance of negative binomial models----
#cannot use deviance(x)
-2*LogLik(M_x)

#zero inflated models-----

#Zero inflated poisson regression----
M <- zeroinfl(x ~ y + ..., data = df_x, family = poisson())

#zero inflated likelihood function----
probability_zero <- 1 / (1 + exp(-(intercept + coefficient_x * x)))

#count likelihood (from zero inflated model)----
probability_count <- 1-(1 / (1 + exp(-(intercept + coefficient_x * x))))

#zero inflated negative binomial regression----
Mzinb <- zeroinfl(x ~ y + ..., dist = 'negbin', data=df_x)

#model comparison (differences of means)----
vuong(M_x, M_y)

#predictions with zero inflated models----
#(poisson in this case), but can be used with 'dist = negbin'
M_zip <- zeroinfl(x ~ y+..., data=df_x)
df_new <- tibble(x = x)
predict(M_zip, newdata = df_new, type='response') # response gives probability

#predictions from coefficients----
#b_x can be used for count and zero parts:
b_zero <- coef(x, model = 'zero')
b_count <- coef(x, model = 'count')

plogis((b_x['(Intercept)'] + b_x['x']*x)) #plogis for probability
exp((b_x['(Intercept)'] + b_x['x']*x)) #exp for mean

#predictions with zero inflated models - for count data----
predict(M_6_zip, newdata = smoking_df_new, type='count')

#predictions with zero inflated models- for the zero inflated part----
#use for when a binary response is 0, e.g. not a smoker
predict(M_6_zip, newdata = smoking_df_new, type='zero')

#zeroinflated poisson predictions using coefficients----
b_count <- coef(M_x, model = 'count')
# logistic regression coefficients
b_zero <- coef(M_x, model = 'zero')
exp(b_count[1] + b_count[2] * x) * (1 - plogis(b_zero[1] + b_zero[2] * x))

#zero inflated poisson predictions of count data----
#i.e. how many x verbed, if x = x
add_predictions(df_x, m_x, type = 'count')

# poisson (zero inflated) predictions of zero (count) data----
add_predictions(df_x, m_x, type = 'zero')
