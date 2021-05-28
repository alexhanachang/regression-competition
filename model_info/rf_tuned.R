#### random forest model ####
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
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_nzv(all_predictors())

recipe %>%
  prep() %>%
  bake(new_data = NULL)


# define-model ------------------------------------------------------------
rf_model <- rand_forest(
  min_n = tune(), 
  mtry = tune()
) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")


# define-tuning-grid ------------------------------------------------------
rf_params <- parameters(rf_model) %>% 
  finalize(fold)

rf_grid <- grid_regular(rf_params, levels = 5)


# define-workflow ---------------------------------------------------------
rf_wflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(recipe)


# tuning ------------------------------------------------------------------
rf_tuned <- rf_wflow %>% 
  tune_grid(resamples = fold, grid = rf_grid)

show_best(rf_tuned, metric = "rmse")

write_rds(rf_tuned, "model_info/rf_tuned.rds")


# tuned-workflow -------------------------------------------------------
rf_wflow_tuned <- rf_wflow %>%
  finalize_workflow(select_best(rf_tuned, metric = "rmse"))


# fit ---------------------------------------------------------------------
rf_results <- fit(rf_wflow_tuned, train)


# test-set-performance ----------------------------------------------------
rf_predictions <- rf_results %>%
  predict(new_data = test) %>% 
  bind_cols(test %>% select(id)) %>%  
  select(id, .pred) %>% 
  rename(Id = id, Predicted = .pred)

write_csv(rf_predictions, "predictions/rf_predictions.csv")
