<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Classifiers/FactorizationMachine.php">[source]</a></span>

# Factorization Machine (Classifier)
A second-order feature interaction model for multi-class classification. One set of FM parameters (global bias, first-order weights, and a latent factor matrix) is maintained per class, and class scores are passed through a **softmax** to produce probabilities. The model is trained online with mini-batch SGD using **cross-entropy** loss. Pairwise interactions are computed in **O(kn)** per sample via the FM identity.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Online](../online.md), [Probabilistic](../probabilistic.md), [Persistable](../persistable.md)

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
use Rubix\ML\Classifiers\FactorizationMachine;

$estimator = new FactorizationMachine(
    factors: 16,
    rate: 0.01,
    l2Penalty: 1e-4,
    epochs: 100,
    batchSize: 64
);
```

## Additional Methods
Return the softmax class probability distribution for each sample:
```php
public proba(Dataset $dataset) : array
```

## Notes
- New classes encountered in `partial()` calls beyond those seen during `train()` are silently ignored. Re-train from scratch if the label set changes.
- Factor vectors are initialised with uniform random values scaled by `1/√k`.
- Binary problems can use `factors = 4–8`; high-cardinality multi-class problems benefit from `factors = 16–32`.

## References
[^1]: S. Rendle. (2010). Factorization Machines.
