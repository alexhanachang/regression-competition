#### random forest model ####
# load-packages -----------------------------------------------------------
library(dplyr)
library(lubridate)
library(tidymodels)
library(tidyverse)

# set-seed -----------------------------------------------------------------
set.seed(123)


# read-in-data ------------------------------------------------------------
test <- read.csv(file = "data/test.csv") 
test[sapply(test, is.character)] <- lapply(test[sapply(test, is.character)], 
                                                       as.factor)

train <- read.csv(file = "data/train.csv")
train[sapply(train, is.character)] <- lapply(train[sapply(train, is.character)], 
                                                         as.factor)

fold <- vfold_cv(train, v = 5, repeats = 3, strata = money_made_inv)


# recipe ------------------------------------------------------------------
recipe <- recipe(money_made_inv ~ ., data = train) %>% 
  step_rm(earliest_cr_line, last_credit_pull_d, purpose, id) %>% 
  step_other(all_nominal_predictors(), -all_outcomes(), threshold = 0.1) %>% 
  step_dummy(all_nominal_predictors(), -all_outcomes(), one_hot = TRUE) %>% 
  step_normalize(all_numeric_predictors(), -all_outcomes()) %>% 
  step_zv(all_predictors(), -all_outcomes()) 

prep(recipe) %>%
  bake(new_data = NULL)

# define-model ------------------------------------------------------------
rf_model <- rand_forest(mode = "regression",
                        min_n = tune(),
                        mtry = tune()) %>% 
  set_engine("ranger")


#set up tuning grid
rf_params <- parameters(rf_model) %>% 
  update(mtry = mtry(range = c(2, 12)))


# define tuning grid
rf_grid <- grid_regular(rf_params, levels = 3)

rf_wflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(recipe)


# workflow ----
rf_tuned <- rf_wflow %>% 
  tune_grid(resamples = fold, 
            grid = rf_grid)

write_rds(rf_tuned, "rf_tuned.rds")

rf_tuned %>% show_best()


#results
rf_wflow_tuned <- rf_wflow %>% 
  finalize_workflow(select_best(rf_tuned, metric = "rmse"))


rf_results <- fit(rf_wflow_tuned, train)

final_rf_results <- rf_results %>%  
  predict(new_data = test) %>%
  bind_cols(test %>% select(id)) %>%
  mutate(Predicted = .pred,
         Id = id) %>%
  select(Id, Predicted)

write_csv(final_rf_results, "rf_predictions_3.csv")






