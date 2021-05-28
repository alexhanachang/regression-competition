#### boosted tree model ####
# load-packages -----------------------------------------------------------
library(tidymodels)
library(tidyverse)


# set-seed -----------------------------------------------------------------
set.seed(123)


# read-in-data ------------------------------------------------------------
train <- read_csv(file = "data/train.csv") %>% janitor::clean_names()
test <- read_csv(file = "data/test.csv") %>% janitor::clean_names()
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
bt_model <- boost_tree(
  mode = "regression",
  mtry = tune(),
  min_n = 3,
  learn_rate = 0.04,
  trees = 500,
  tree_depth = 5) %>% 
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
bt_tuned <- bt_wflow %>% 
  tune_grid(resamples = fold, grid = bt_grid)

show_best(bt_tuned, metric = "rmse")

write_rds(bt_tuned, "model_info/bt_tuned_3.rds")


# tuned-workflow -------------------------------------------------------
bt_wflow_tuned <- bt_wflow %>%
  finalize_workflow(select_best(bt_tuned, metric = "rmse"))


# fit ---------------------------------------------------------------------
bt_results <- fit(bt_wflow_tuned, train)


# test-set-performance ----------------------------------------------------
final_bt_results <- bt_results %>%
  predict(new_data = test) %>%
  bind_cols(test %>% select(id)) %>% 
  select(id, .pred) %>% 
  rename(Id = id, Predicted = .pred)

write_csv(final_bt_results, "predictions/bt_predictions5.csv")
