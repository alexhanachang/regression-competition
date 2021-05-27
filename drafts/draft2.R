library(tidyverse)
library(tidymodels)
library(dplyr)
library(lubridate)

#loading data and making the variables factors
money_test <- read.csv(file = "data/test.csv") 
money_test[sapply(money_test, is.character)] <- lapply(money_test[sapply(money_test, is.character)], 
                                                       as.factor)

money_train <- read.csv(file = "data/train.csv")
money_train[sapply(money_train, is.character)] <- lapply(money_train[sapply(money_train, is.character)], 
                                                         as.factor)

money_train <- money_train %>% 
  select(-c(earliest_cr_line, last_credit_pull_d))

#make outcome a factor and make dates


set.seed(405)

#looking at data 
skimr::skim_without_charts(money_train)
#no missingness!
money_train

#recipe
money_rf_recipe <- recipe(money_made_inv ~ ., data = money_train) %>% 
  step_rm(purpose, id) %>% 
  step_other(all_nominal(), -all_outcomes(), threshold = 0.1) %>% 
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>% 
  step_normalize(all_numeric(), -all_outcomes()) %>% 
  step_zv(all_predictors(), -all_outcomes()) 

#cross validation folds
money_folds <- vfold_cv(data = money_train, v = 5, repeats = 3, strata = money_made_inv)

prep(money_rf_recipe) %>% 
  bake(new_data = NULL)

#define model
rf_model <- rand_forest(mode = "regression",
                        min_n = tune(),
                        mtry = tune()) %>% 
  set_engine("ranger")

#set up tuning grid
rf_params <- parameters(rf_model) %>% 
  update(mtry = mtry(range = c(2, 12)))


# define tuning grid
rf_grid <- grid_regular(rf_params, levels = 3)

rf_workflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(money_rf_recipe)



# workflow ----
rf_tuned <- rf_workflow %>% 
  tune_grid(resamples = money_folds, 
            grid = rf_grid)

write_rds(rf_tuned, "model_info/rf_results.rds")




