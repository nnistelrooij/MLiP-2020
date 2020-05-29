import lightgbm as lgb

# All available parameters can be found here:  
# https://lightgbm.readthedocs.io/en/latest/Parameters.html
default_params = {
                  "objective" : "poisson",
                  "metric" :"rmse",
                  "force_row_wise" : True,
                  "learning_rate" : 0.075,
                  "sub_row" : 0.75,
                  "bagging_freq" : 1,
                  "lambda_l2" : 0.1,
                  'num_leaves': 128,
                  "min_data_in_leaf": 100,
                }

def train(train_set, val_set, num_rounds=100, early_stopping_rounds=10, 
          params=default_params, save_model=None):
  """
  Train LightGBM model.  
  The `params` are adapted from:
  https://www.kaggle.com/kneroma/m5-first-public-notebook-under-0-50

  Returns [Booster]:
    Trained model.
  """

  model = lgb.train(params, 
                    train_set, 
                    num_boost_round=num_rounds,
                    valid_sets=[train_set, val_set], 
                    valid_names=['train', 'validation'],
                    categorical_feature=train_set.categorical_feature,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=int(num_rounds/10))
  
  if save_model:
    model.save_model(save_model)

  return model
