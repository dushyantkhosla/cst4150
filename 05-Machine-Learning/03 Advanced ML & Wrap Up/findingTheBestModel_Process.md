
# 1. We're trying to build models that neither Overfit or Underfit

# 2. Overfitting/Underfitting is directly affected by the model's complexity

# A highly complex model might overfit. (High Variance)
# A too simple model might underfit. (High Bias)

# 3. Model Complexity is controlled by its Hyperparamters.

# 4. So, I have a SEARCH PROBLEM at hand.

# 5. I want to find the optimum combination of model hyperparameters that neither overfit nor underfit the data.

# 6. The only way to do this is with Trial and Error.

# 7. So, we set up a grid containing all possible (reasonable) combinations of Hyperparameter values.

# 8. We then fit models to each combination, and find the (cross validated) Performance of each combination of hyperparamter values.

# 9. GridSearchCV *automates* this process for me. All I need to specify are - 
# the model to use, 
# the Grid, 
# the performance metric to use (depending on the model), 
# the crossvalidation scheme to use (3- or 5- or 10- fold CV)

# 10. GridSearchCV finds the Best Model that optimizes the Performance Metric.
