## 4th Place Solution — Hull Tactical Market Prediction
First of all, thanks to the organizers for designing such an interesting competition. I particularly enjoyed that success depended as much on portfolio construction and risk management as on prediction itself.

This solution is somewhat unusual compared to a typical Kaggle workflow. There is **no machine learning model**, only walk-forward validation and almost no hyperparameter optimization. Instead, the final submission consists of a simple rule-based alpha combined with a fairly sophisticated portfolio construction framework.

The alpha itself predates this competition, so I won't discuss its exact construction, but I hope the portfolio engineering ideas are useful to others.

## High-level philosophy
Very early on I stopped thinking of the competition as a forecasting problem.

Instead, I viewed it as solving
$$ \text{Portfolio} = \text{Aplha} + \text{Risk Management}. $$

The evaluation metric strongly rewards maintaining market-like returns while penalizing excessive volatility. Under this scoring function, improving portfolio construction often contributes more than marginal improvements in predictive accuracy.

This observation guided almost every design decision.

Instead of asking
> "Can I predict tomorrow's return better?"

I found it more useful to ask
> Given the alpha I already have, what is the best possible portfolio I can build?"

## Starting point
Unlike many participants, I did not search for alpha inside the competition data.

The core signal had been researched before the competition as part of my personal quantitative research. It belongs to the family of short-horizon mean-reversion indicators, and had already been validated outside the Kaggle environment.

Because of that, I deliberately avoided tuning the signal specifically for the competition.

Only a handful of complete experiments were run during the entire competition.

In my experience, once a signal already has a convincing economic rationale and survives multiple market environments, additional optimization is much more likely to fit noise than improve genuine performance.

## Portfolio construction
Most of my work actually went into transforming a relatively sparse alpha signal into a robust allocation process.

This ended up being considerably more important than improving the alpha itself.

### Inverse-volatility weighting
The primary alpha is naturally sparse. During long periods it may remain inactive, producing an unstable estimate of realized portfolio volatility.

To stabilize the resulting allocation, I combined it with a few additional low-alpha signals whose objective was not to improve prediction but to provide a smoother exposure profile.

Rather than fitting weights through regression or optimization, I used inverse-volatility weighting.

For each signal (i),
$$ w_i = \frac{\sigma_i^{-1}}{\Sigma_j\sigma_j^{-1}} $$
where
$$ \sigma_i $$
is the rolling volatility of signal (i).

This has several desirable properties:

- no optimization problem to solve;
- no covariance matrix estimation;
- automatic downweighting of unstable signals;
- adaptive allocations across different market regimes.

While simple, this approach produced remarkably stable allocations.

An interesting consequence is that the strongest alpha signal does not necessarily receive the largest portfolio weight. Since it is relatively sparse, its standalone volatility is low, and inverse-vol weighting naturally balances it against the more persistent auxiliary components.

The additional signals should therefore be viewed primarily as portfolio stabilizers rather than independent alpha sources.

## Volatility targeting
The second major component was a volatility targeting overlay.

After combining the signals, the portfolio exposure is rescaled to target a desired level of realized volatility.

Conceptually,

$$ L_t=\frac{\sigma_\text{target}}{\hat{\sigma}_\text{portfolio, t}} $$
where
$$ (L_t) $$
is the leverage multiplier,
$$ (\hat{\sigma}_\text{portfolio, t}) $$
is the estimated realized portfolio volatility.

The final allocation becomes
$$ w_t^{\text{final}} = \text{clip}(L_t w_t, 0, 2)  $$
where clipping enforces the competition's leverage constraints.

The volatility estimate is intentionally computed over a relatively long window and only updated periodically, preventing the leverage from reacting excessively to short-lived market noise.

I found that this overlay contributed more to the final leaderboard score than almost any modification to the alpha itself.

## Why not optimize weights?
One natural question is why I did not solve something like
$$ \max_w \frac{\mu^T w}{\sqrt{w^T \Sigma w}} $$

There were two reasons.

First, the number of signals is very small, making a fully optimized solution unnecessarily complex.

Second, estimating covariance matrices from relatively short financial time series introduces estimation error that often dominates the theoretical gains from optimization.

Inverse-volatility weighting is well known to be an excellent compromise between robustness and performance.

In practice, I found it more reliable than more sophisticated allocation schemes.

## Feature engineering
The competition provided a fairly rich feature set.

Rather than manually inspecting every feature, I applied the same transformation pipeline to all candidates before evaluating them.

Most engineered features were discarded.

Only a very small number survived into the final portfolio, and even those were selected primarily because they improved the stability of the allocation process rather than because they generated large standalone returns.

This is probably one of the biggest lessons I learned during the competition:
> A feature does not need to predict returns to improve a trading strategy.

Sometimes its greatest contribution is reducing estimation noise elsewhere in the pipeline.

## Simplicity over optimization
One aspect I'm happiest with is how little tuning the final solution required.

There was:

- no grid search;
- no Bayesian optimization;
- no cross-validation;
- no ensembling (retrospectively this would have been a good idea: Ensembling slight variations of the signal. But I didn't think of it at that time).

The entire solution is essentially a deterministic pipeline with a small number of parameters inherited from prior research.

This greatly reduced the risk of overfitting to the historical data.

## Limitations
The strategy is built around a short-term mean-reverting alpha.

Consequently, it is expected to underperform during persistent low-volatility trending markets while performing relatively better during volatile or range-bound environments.

This is not a weakness specific to the competition—it is an inherent property of this class of strategies.

You can see in equity curves below, that the strategy does not work very well in Bull Markets (Nothing usually beats the market in bull markets).

## Performance Summary
| Configuration                                       | Total Return | Ann. Return | Ann. Volatility | Sharpe | Max Drawdown | Hit Rate |
|-----------------------------------------------------|--------------|-------------|-----------------|--------|--------------|----------|
| Stage A — Vanilla signal (no volatility targeting)  | 258.7%       | 5.2%        | 7.7%            | 0.68   | -13.6%       | 59.6%    |
| Stage B — Competition signal + volatility targeting | 776.9%       | 9.4%        | 14.3%           | 0.66   | -25.8%       | 59.6%    |
| Buy & Hold Benchmark                                | 215.4%       | 6.3%        | 19.2%           | 0.33   | -63.8%       | 53.9%    |

## Takeaways
This competition reinforced several ideas that I already believed:

- Good portfolio construction can matter more than marginal improvements in forecasting.
- Simple, interpretable models are often harder to overfit.
- Existing domain knowledge is a powerful regularizer.
- Robust risk management can create more value than adding increasingly sophisticated predictive models.

Overall, I think this competition rewarded disciplined quantitative engineering more than predictive complexity, which made it especially enjoyable.

Thanks again to the organizers, and congratulations to everyone who participated. I'm looking forward to reading the other solution write-ups.