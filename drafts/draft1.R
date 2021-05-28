# load-packages -----------------------------------------------------------
library(glmnet)
library(lubridate)
library(skimr)
library(tidymodels)
library(tidyverse)


# read-in-data ------------------------------------------------------------
test <- read.csv(file = "data/test.csv") 
train <- read.csv(file = "data/train.csv")

train <- train %>% 
  mutate_if(is_character, as_factor) %>% 
  select(-last_credit_pull_d, -earliest_cr_line, -emp_title)


# set-seed -----------------------------------------------------------------
set.seed(2021)


# quality-check -----------------------------------------------------------
skim_without_charts(train)


# recipe ------------------------------------------------------------------
rf_recipe <- recipe(money_made_inv ~ ., data = train) %>% 
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>%
  step_normalize(all_numeric()) 

rf_recipe %>% 
  prep() %>% 
  bake(new_data = NULL)


# fold-data ---------------------------------------------------------------
folds <- vfold_cv(data = train, v = 5, repeats = 3, strata = money_made_inv)


# define-model ------------------------------------------------------------
rf_model <- rand_forest(
  mtry = tune(), 
  min_n = tune()) %>% 
  set_mode("regression") %>% 
  set_engine("ranger")


# define-tuning-grid ------------------------------------------------------
rf_params <- parameters(rf_model) %>% 
  update(
    mtry = mtry(range = c(2, 12))
  )

rf_grid <- grid_regular(rf_params, levels = 5)


# define-workflow ---------------------------------------------------------
rf_wflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(rf_recipe)


# tuning ------------------------------------------------------------------
rf_tuned <- rf_wflow %>% 
  tune_grid(folds, grid = rf_grid)

write_rds(rf_tuned, "model_info/rf_tuned.rds")

