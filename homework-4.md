---
title: "Homework 4"
author: "PSTAT 131/231"
output:
    html_document:
      toc: true
      toc_float: true
      code_folding: show
      keep_md: true
---



## Resampling

For this assignment, we will continue working with part of a [Kaggle data set](https://www.kaggle.com/c/titanic/overview) that was the subject of a machine learning competition and is often used for practicing ML models. The goal is classification; specifically, to predict which passengers would survive the [Titanic shipwreck](https://en.wikipedia.org/wiki/Titanic).

![Fig. 1: RMS Titanic departing Southampton on April 10, 1912.](RMS_Titanic.jpg){width="363"}

Load the data from `data/titanic.csv` into *R* and familiarize yourself with the variables it contains using the codebook (`data/titanic_codebook.txt`).

Notice that `survived` and `pclass` should be changed to factors. When changing `survived` to a factor, you may want to reorder the factor so that *"Yes"* is the first level.

Make sure you load the `tidyverse` and `tidymodels`!

*Remember that you'll need to set a seed at the beginning of the document to reproduce your results.*

Create a recipe for this dataset **identical** to the recipe you used in Homework 3.

### Question 1

Split the data, stratifying on the outcome variable, `survived.`  You should choose the proportions to split the data into. Verify that the training and testing data sets have the appropriate number of observations. 

```r
set.seed(3435)
titanic_split <- initial_split(titanic, prop = 0.70, strata = survived)
titanic_train <- training(titanic_split)
titanic_test <- testing(titanic_split)

dim(titanic_train)
```

```
## [1] 623  12
```

```r
dim(titanic_test)
```

```
## [1] 268  12
```

```r
titanic_train_recipe <- recipe(survived ~ pclass + sex + age + sib_sp + parch + fare, data = titanic_train) %>%
  step_impute_linear(age, impute_with = imp_vars(sib_sp)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_interact(~ starts_with("sex"):fare + age:fare)
```

### Question 2

Fold the **training** data. Use *k*-fold cross-validation, with $k = 10$.

```r
lm_spec <- linear_reg() %>%
  set_mode("regression") %>%
  set_engine("lm")
```

```r
set.seed(345)
titanic_folds <- vfold_cv(titanic_train, v = 10)
```


### Question 3

In your own words, explain what we are doing in Question 2. What is *k*-fold cross-validation? Why should we use it, rather than simply fitting and testing models on the entire training set? If we **did** use the entire training set, what resampling method would that be?

Answer:
K-fold cross-validation is a data partitioning strategy so that allows you to more effectively use your data set to build a more generalized model. We use k-fold cross-validation because it ensures that every observation from the original data set has the chance of appearing in the training and the test set. If we used the entire training set, that would be the validation set approach.

### Question 4

Set up workflows for 3 models:

1. A logistic regression with the `glm` engine;
2. A linear discriminant analysis with the `MASS` engine;
3. A quadratic discriminant analysis with the `MASS` engine.

How many models, total, across all folds, will you be fitting to the data? To answer, think about how many folds there are, and how many models you'll fit to each fold.


```r
log_reg <- logistic_reg() %>% 
  set_engine("glm") %>% 
  set_mode("classification")

log_wkflow <- workflow() %>% 
  add_model(log_reg) %>% 
  add_recipe(titanic_train_recipe)

lda_mod <- discrim_linear() %>% 
  set_mode("classification") %>% 
  set_engine("MASS")

lda_wkflow <- workflow() %>% 
  add_model(lda_mod) %>% 
  add_recipe(titanic_train_recipe)

qda_mod <- discrim_quad() %>% 
  set_mode("classification") %>% 
  set_engine("MASS")

qda_wkflow <- workflow() %>% 
  add_model(qda_mod) %>% 
  add_recipe(titanic_train_recipe)
```
There will be 30 models to fit to the data.

### Question 5

Fit each of the models created in Question 4 to the folded data.

**IMPORTANT:** *Some models may take a while to run – anywhere from 3 to 10 minutes. You should NOT re-run these models each time you knit. Instead, run them once, using an R script, and store your results; look into the use of [loading and saving](https://www.r-bloggers.com/2017/04/load-save-and-rda-files/). You should still include the code to run them when you knit, but set `eval = FALSE` in the code chunks.*

```r
log_fit <- fit_resamples(log_wkflow, titanic_folds)
log_fit
```

```
## # Resampling results
## # 10-fold cross-validation 
## # A tibble: 10 x 4
##    splits           id     .metrics         .notes          
##    <list>           <chr>  <list>           <list>          
##  1 <split [560/63]> Fold01 <tibble [2 x 4]> <tibble [0 x 3]>
##  2 <split [560/63]> Fold02 <tibble [2 x 4]> <tibble [0 x 3]>
##  3 <split [560/63]> Fold03 <tibble [2 x 4]> <tibble [0 x 3]>
##  4 <split [561/62]> Fold04 <tibble [2 x 4]> <tibble [0 x 3]>
##  5 <split [561/62]> Fold05 <tibble [2 x 4]> <tibble [0 x 3]>
##  6 <split [561/62]> Fold06 <tibble [2 x 4]> <tibble [0 x 3]>
##  7 <split [561/62]> Fold07 <tibble [2 x 4]> <tibble [0 x 3]>
##  8 <split [561/62]> Fold08 <tibble [2 x 4]> <tibble [0 x 3]>
##  9 <split [561/62]> Fold09 <tibble [2 x 4]> <tibble [0 x 3]>
## 10 <split [561/62]> Fold10 <tibble [2 x 4]> <tibble [0 x 3]>
```

```r
qda_fit <- fit_resamples(qda_wkflow, titanic_folds)
qda_fit
```

```
## # Resampling results
## # 10-fold cross-validation 
## # A tibble: 10 x 4
##    splits           id     .metrics         .notes          
##    <list>           <chr>  <list>           <list>          
##  1 <split [560/63]> Fold01 <tibble [2 x 4]> <tibble [0 x 3]>
##  2 <split [560/63]> Fold02 <tibble [2 x 4]> <tibble [0 x 3]>
##  3 <split [560/63]> Fold03 <tibble [2 x 4]> <tibble [0 x 3]>
##  4 <split [561/62]> Fold04 <tibble [2 x 4]> <tibble [0 x 3]>
##  5 <split [561/62]> Fold05 <tibble [2 x 4]> <tibble [0 x 3]>
##  6 <split [561/62]> Fold06 <tibble [2 x 4]> <tibble [0 x 3]>
##  7 <split [561/62]> Fold07 <tibble [2 x 4]> <tibble [0 x 3]>
##  8 <split [561/62]> Fold08 <tibble [2 x 4]> <tibble [0 x 3]>
##  9 <split [561/62]> Fold09 <tibble [2 x 4]> <tibble [0 x 3]>
## 10 <split [561/62]> Fold10 <tibble [2 x 4]> <tibble [0 x 3]>
```

```r
lda_fit <- fit_resamples(lda_wkflow, titanic_folds)
lda_fit
```

```
## # Resampling results
## # 10-fold cross-validation 
## # A tibble: 10 x 4
##    splits           id     .metrics         .notes          
##    <list>           <chr>  <list>           <list>          
##  1 <split [560/63]> Fold01 <tibble [2 x 4]> <tibble [0 x 3]>
##  2 <split [560/63]> Fold02 <tibble [2 x 4]> <tibble [0 x 3]>
##  3 <split [560/63]> Fold03 <tibble [2 x 4]> <tibble [0 x 3]>
##  4 <split [561/62]> Fold04 <tibble [2 x 4]> <tibble [0 x 3]>
##  5 <split [561/62]> Fold05 <tibble [2 x 4]> <tibble [0 x 3]>
##  6 <split [561/62]> Fold06 <tibble [2 x 4]> <tibble [0 x 3]>
##  7 <split [561/62]> Fold07 <tibble [2 x 4]> <tibble [0 x 3]>
##  8 <split [561/62]> Fold08 <tibble [2 x 4]> <tibble [0 x 3]>
##  9 <split [561/62]> Fold09 <tibble [2 x 4]> <tibble [0 x 3]>
## 10 <split [561/62]> Fold10 <tibble [2 x 4]> <tibble [0 x 3]>
```

### Question 6

Use `collect_metrics()` to print the mean and standard errors of the performance metric *accuracy* across all folds for each of the four models.

Decide which of the 3 fitted models has performed the best. Explain why. *(Note: You should consider both the mean accuracy and its standard error.)*

```r
collect_metrics(log_fit)
```

```
## # A tibble: 2 x 6
##   .metric  .estimator  mean     n std_err .config             
##   <chr>    <chr>      <dbl> <int>   <dbl> <fct>               
## 1 accuracy binary     0.786    10  0.0152 Preprocessor1_Model1
## 2 roc_auc  binary     0.830    10  0.0135 Preprocessor1_Model1
```

```r
collect_metrics(lda_fit)
```

```
## # A tibble: 2 x 6
##   .metric  .estimator  mean     n std_err .config             
##   <chr>    <chr>      <dbl> <int>   <dbl> <fct>               
## 1 accuracy binary     0.780    10  0.0134 Preprocessor1_Model1
## 2 roc_auc  binary     0.833    10  0.0117 Preprocessor1_Model1
```

```r
collect_metrics(qda_fit)
```

```
## # A tibble: 2 x 6
##   .metric  .estimator  mean     n std_err .config             
##   <chr>    <chr>      <dbl> <int>   <dbl> <fct>               
## 1 accuracy binary     0.762    10  0.0153 Preprocessor1_Model1
## 2 roc_auc  binary     0.827    10  0.0136 Preprocessor1_Model1
```
The lda model performed the best because it has the lowest standard error and the second highest accuracy.

### Question 7

Now that you’ve chosen a model, fit your chosen model to the entire training dataset (not to the folds).

```r
lda_fit1 <- fit(lda_wkflow, titanic_train)
lda_fit1
```

```
## == Workflow [trained] ==========================================================
## Preprocessor: Recipe
## Model: discrim_linear()
## 
## -- Preprocessor ----------------------------------------------------------------
## 3 Recipe Steps
## 
## * step_impute_linear()
## * step_dummy()
## * step_interact()
## 
## -- Model -----------------------------------------------------------------------
## Call:
## lda(..y ~ ., data = data)
## 
## Prior probabilities of groups:
##       Yes        No 
## 0.3836276 0.6163724 
## 
## Group means:
##          age    sib_sp     parch     fare pclass_X2 pclass_X3  sex_male
## Yes 29.16496 0.4393305 0.4267782 48.81424 0.2552301 0.3389121 0.3472803
## No  30.05569 0.5807292 0.3463542 22.99299 0.1744792 0.6770833 0.8489583
##     sex_male_x_fare fare_x_age
## Yes        12.91229  1558.8148
## No         19.63018   697.5167
## 
## Coefficients of linear discriminants:
##                           LD1
## age              0.0322121339
## sib_sp           0.2208266971
## parch            0.0918861052
## fare             0.0028507145
## pclass_X2        0.8006495804
## pclass_X3        1.5603350324
## sex_male         1.9823061521
## sex_male_x_fare  0.0017471177
## fare_x_age      -0.0001416796
```

### Question 8

Finally, with your fitted model, use `predict()`, `bind_cols()`, and `accuracy()` to assess your model’s performance on the testing data!

Compare your model’s testing accuracy to its average accuracy across folds. Describe what you see.

```r
lda_acc <- augment(lda_fit1, new_data = titanic_test) %>%
  accuracy(truth = survived, estimate = .pred_class)

bind_rows(collect_metrics(lda_fit)[1,1:3], lda_acc)
```

```
## # A tibble: 2 x 4
##   .metric  .estimator   mean .estimate
##   <chr>    <chr>       <dbl>     <dbl>
## 1 accuracy binary      0.780    NA    
## 2 accuracy binary     NA         0.817
```
The estimate for the testing data is much higher than the average of the 10-fold training data.

## Required for 231 Students

Consider the following intercept-only model, with $\epsilon \sim N(0, \sigma^2)$:

$$
Y=\beta+\epsilon
$$

where $\beta$ is the parameter that we want to estimate. Suppose that we have $n$ observations of the response, i.e. $y_{1}, ..., y_{n}$, with uncorrelated errors.

### Question 9

Derive the least-squares estimate of $\beta$.

$$
Let \ \ \hat{Y}=\beta_{0}+\beta_{1} X,\ 
and\ 
\begin{aligned}
\hat{\beta}_{1} &=\frac{\sum_{i=1}^{n}\left(X_{i}-\bar{X}\right)\left(Y_{i}-\bar{Y}\right)}{\sum_{i=1}^{n}\left(X_{i}-\bar{X}\right)^{2}} \\
\hat{\beta}_{0} &=\bar{Y}-\hat{\beta}_{1} \bar{X}
\end{aligned}\\
If\ \ \hat{Y}=\beta_{0}+0 X,\\
then\  \hat{Y}=\beta_{0}
$$

### Question 10

Suppose that we perform leave-one-out cross-validation (LOOCV). Recall that, in LOOCV, we divide the data into $n$ folds. What is the covariance between $\hat{\beta}^{(1)}$, or the least-squares estimator of $\beta$ that we obtain by taking the first fold as a training set, and $\hat{\beta}^{(2)}$, the least-squares estimator of $\beta$ that we obtain by taking the second fold as a training set?

The covariance would be close to 1 because the betas of each fold are the average of the rest of the folds.
