<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Regressors/ElasticNet.php">[source]</a></span>

# Elastic Net
A linear regressor that combines L1 (*Lasso*) and L2 (*Ridge*) regularization into a single penalty term. The mixing between the two is controlled by *l1Ratio* — a value of `1.0` gives pure Lasso (sparse solutions), `0.0` gives pure Ridge, and intermediate values give the elastic net effect. The model is trained using coordinate descent with an in-place residual update, giving **O(np)** work per epoch.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Ranks Features](../ranks-features.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | alpha | 1.0 | float | The combined regularization strength (≥ 0). |
| 2 | l1Ratio | 0.5 | float | Mixing ratio between L1 and L2 (0 = Ridge, 1 = Lasso). |
| 3 | epochs | 1000 | int | Maximum number of coordinate descent iterations. |
| 4 | tol | 1e-4 | float | Convergence tolerance on the maximum weight change per epoch. |

## Example
```php
use Rubix\ML\Regressors\ElasticNet;

$estimator = new ElasticNet(
    alpha: 0.5,
    l1Ratio: 0.7,   // 70% L1 (Lasso) + 30% L2 (Ridge)
    epochs: 2000,
    tol: 1e-5
);
```

## Additional Methods
Return the learned weight coefficients or `null` if untrained:
```php
public coefficients() : float[]|null
```

Return the learned bias (intercept) or `null` if untrained:
```php
public bias() : float|null
```

Return the absolute-value feature importances:
```php
public featureImportances() : float[]
```

## Notes
- When `alpha = 0` the model reduces to ordinary least squares (OLS).
- Setting `l1Ratio = 1` recovers Lasso; `l1Ratio = 0` recovers Ridge.
- Coordinate descent convergences faster than gradient descent for this problem and requires no learning-rate tuning.

## References
[^1]: H. Zou et al. (2005). Regularization and Variable Selection via the Elastic Net.
