This was my first Kaggle silver medal, and reaching this goal feels especially rewarding after years of learning from the work, notebooks and discussions shared by the Kaggle community.

It was also my first competition in which the leaderboard evolved through monthly batches of genuinely unseen live data. The model was not evaluated only on a hidden historical split; it had to operate while the global economy and financial markets were moving through a particularly challenging and uncertain period.

This solution was built around a simple idea:
> In a noisy financial problem, accurately predicting the exact return may be unrealistic. It may be more useful to identify whether the expected opportunity is relatively positive or negative—and make a conservative decision from there.

## 1. Adding temporal context
The dataset contained many anonymised market, volatility, sentiment and macroeconomic features.

Rather than creating historical transformations for every column which would be impossible to do, I selected 14 promising features and added:

Lags of 1, 3, 5, 7, 14 and 20 periods \
Rolling means over 2, 5, 10, 20 and 60 periods \
Rolling standard deviations over the same windows 

This produced 224 additional temporal features.

The intention was to preserve the breadth of the original dataset while adding deeper historical context only where it appeared most useful.

A feature’s current value may not be informative by itself. Its meaning can depend on whether it is rising, stable or unusually volatile compared with its recent history.

## 2. Using one LightGBM model
This version used a single LightGBM regression model.

I did try to experiment with producing ensemble model using Catboost, Xgboost along with LightGBM but test scores were consistently lower than this simpler version showing hints that Occam's Razor finds a very good application for this uncertain in nature problem.

Single model complexity was dealth through feature engineering and hyperparameter tuning.

The model was tuned using four-fold time-series cross-validation. Chronological validation was essential because random splits would not represent how the model would operate on future market observations.

## 3. Optimising ranking rather than exact return magnitude
Although LightGBM used RMSE for early stopping, Optuna selected hyperparameters based on the average Spearman rank correlation across the time-series folds.

The model was intentionally interested in consistently ranking correctly strong and weak future return opportunities rather than predicting the exact return on point.

## 4. The unexpected simplicity of the final allocation
Although the competition allowed allocations between 0 and 2, I intentionally restricted the strategy to two possible actions:

0: move to the risk-free asset \
1: maintain normal market exposure

The binary policy acts as a form of regularisation. It avoids treating small differences between uncertain predictions as meaningful, removes the need to calibrate leverage, and limits the impact of large forecasting errors.

I did experiment with dynamic allocations between 0 and 2 but Leaderboard scores were consistently lower.

## 5. Lessons learned
This experience had several lessons for me.

First, strong offline validation is necessary, but it cannot fully reproduce the uncertainty of future deployment.

Second, in noisy financial problems, robustness can be more valuable than complexity. Large ensembles and highly sensitive allocation functions may improve historical scores, but they can also amplify errors when the future behaves differently from the past.

Also, this competition reminded me that improvement on Kaggle is cumulative. Each notebook studied, low scoring submission and shared community insight contributes to the next solution. So this might be an important milestone for me, but it is also evidence of how much I have learned from others—and how much there is still left to learn.

Finally, I would like to sincerely thank the organizers for creating a competition that challenged not only our modelling skills, but also our assumptions about uncertainty, risk and what it means for a solution to generalize in a live forecasting problem.