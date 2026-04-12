<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Regressors/FactorizationMachine.php">[source]</a></span>

# Factorization Machine (Regressor)
A second-order feature interaction model that captures pairwise interactions between all features through a low-rank factorization. The key advantage over a degree-2 polynomial kernel is that interactions are computed in **O(kn)** time (where *k* is the number of latent factors) rather than O(n²), via the FM identity:

```
Σ_{i<j} ⟨v_i, v_j⟩ x_i x_j = ½ Σ_f [(Σ_i v_{if} x_i)² − Σ_i v²_{if} x²_i]
```

The model is trained online with mini-batch SGD using mean squared error as the loss function.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Online](../online.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | factors | 10 | int | Number of latent factors k per feature. |
| 2 | rate | 0.01 | float | Initial SGD learning rate. |
| 3 | l2Penalty | 1e-4 | float | L2 regularization strength on weights and factors. |
| 4 | epochs | 100 | int | Training epochs per `train()` or `partial()` call. |
| 5 | batchSize | 64 | int | Samples per mini-batch. |

## Example
```php
use Rubix\ML\Regressors\FactorizationMachine;

$estimator = new FactorizationMachine(
    factors: 16,
    rate: 0.005,
    l2Penalty: 1e-4,
    epochs: 200,
    batchSize: 128
);
```

## Notes
- Factor vectors are initialised with uniform random values scaled by `1/√k`.
- Use `partial()` for streaming / incremental training.
- Increasing `factors` captures richer interactions at the cost of more parameters.

## References
[^1]: S. Rendle. (2010). Factorization Machines.
