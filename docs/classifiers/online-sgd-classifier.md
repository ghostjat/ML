<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Classifiers/OnlineSGDClassifier.php">[source]</a></span>

# Online SGD Classifier
A multi-class linear classifier trained incrementally via Stochastic Gradient Descent. One linear weight vector is maintained per class (OvR — one-vs-rest). Step sizes are adapted per feature using **AdaGrad**, making the classifier well-suited for high-dimensional or sparse data.

Three loss functions are supported:

| Loss | Description |
|------|-------------|
| `log` | Logistic (cross-entropy). Outputs calibrated probabilities. |
| `hinge` | Linear SVM hinge loss. Maximises the margin. |
| `perceptron` | Classic perceptron update. Zero gradient when correct. |

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Online](../online.md), [Probabilistic](../probabilistic.md), [Verbose](../verbose.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | rate | 0.01 | float | Initial AdaGrad learning rate. |
| 2 | l2Penalty | 1e-4 | float | L2 regularization strength. |
| 3 | loss | 'log' | string | Loss function: `log`, `hinge`, or `perceptron`. |
| 4 | epochs | 100 | int | Training epochs per `train()` or `partial()` call. |
| 5 | batchSize | 64 | int | Samples per mini-batch. |

## Example
```php
use Rubix\ML\Classifiers\OnlineSGDClassifier;

$estimator = new OnlineSGDClassifier(
    rate: 0.01,
    l2Penalty: 1e-4,
    loss: 'hinge',
    epochs: 200,
    batchSize: 128
);

// Incremental training on data streams:
$estimator->train($batch1);
$estimator->partial($batch2);
$estimator->partial($batch3);
```

## Additional Methods
Return the per-class weight vectors:
```php
public weights() : array|null
```

Return softmax class probabilities (regardless of loss function):
```php
public proba(Dataset $dataset) : array
```

## Notes
- With `hinge` and `perceptron` losses, `proba()` returns softmax-normalised scores, not true probabilities.
- AdaGrad accumulators are preserved between `partial()` calls, so the effective learning rate decays over time.
- For true probabilistic outputs, use `loss: 'log'`.

## References
[^1]: J. Duchi et al. (2011). Adaptive Subgradient Methods for Online Learning and Stochastic Optimization.
