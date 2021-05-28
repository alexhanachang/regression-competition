#### boosted tree model ####
# load-packages -----------------------------------------------------------
library(janitor)
library(tidymodels)
library(tidyverse)


# set-seed -----------------------------------------------------------------
set.seed(123)


# read-in-data ------------------------------------------------------------
train <- read_csv(file = "data/train.csv") %>% clean_names()
test <- read_csv(file = "data/test.csv") %>% clean_names()
fold <- vfold_cv(train, v = 5, repeats = 3, strata = money_made_inv)


# recipe ------------------------------------------------------------------
recipe <- recipe(money_made_inv ~ ., data = train) %>% 
  step_rm(earliest_cr_line, emp_title, emp_length, delinq_amnt, id, last_credit_pull_d, purpose) %>% 
  step_dummy(all_nominal(), one_hot = TRUE) %>% 
  step_normalize(all_numeric(), -all_outcomes()) %>% 
  step_nzv(all_predictors())

recipe %>%
  prep() %>%
  bake(new_data = NULL)

save(train, test, fold, recipe, file = "model_info/bt_setup.rda")


# define-model ------------------------------------------------------------
bt_model <- boost_tree(
  mode = "regression",
  mtry = tune(),
  min_n = 3,
  learn_rate = 0.04,
  trees = 500,
  tree_depth = 5,
  sample_size = 0.65) %>% 
  set_engine("xgboost")


# define-tuning-grid ------------------------------------------------------
bt_params <- parameters(bt_model) %>% 
  update(mtry = mtry(range(70, 80)))

bt_grid <- grid_regular(bt_params, levels = 3)


# define-workflow ---------------------------------------------------------
bt_wflow <- workflow() %>% 
  add_model(bt_model) %>% 
  add_recipe(recipe)


# tuning ------------------------------------------------------------------
bt_tuned <- bt_workflow %>% 
  tune_grid(resamples = fold, grid = bt_grid)

show_best(bt_tuned, metric = "rmse")

# A tibble: 3 x 7
# mtry .metric .estimator  mean     n std_err .config             
# <int> <chr>   <chr>      <dbl> <int>   <dbl> <chr>               
# 1    75 rmse    standard    509.    15    15.2 Preprocessor1_Model2
# 2    70 rmse    standard    510.    15    14.8 Preprocessor1_Model1
# 3    80 rmse    standard    516.    15    15.9 Preprocessor1_Model3

write_rds(bt_tuned, "model_info/bt_tuned.rds")


# tuned-workflow -------------------------------------------------------
bt_workflow_tuned <- bt_workflow %>%
  finalize_workflow(select_best(bt_tuned, metric = "rmse"))


# fit ---------------------------------------------------------------------
bt_results <- fit(bt_workflow_tuned, train)
bt_results

# test-set-performance ----------------------------------------------------
bt_results %>%
  predict(new_data = test) 
  bind_cols(test %>% select(id)) %>%  
  select(id, .pred) %>% 
  rename(Id = id, Predicted = .pred)

write_csv(bt_predictions, "predictions/bt_predictions.csv")
