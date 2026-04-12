<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Regressors/GaussianProcessRegressor.php">[source]</a></span>

# Gaussian Process Regressor
A non-parametric Bayesian regressor that models the posterior distribution over functions using a Gaussian process prior. The **RBF (squared-exponential) kernel** measures similarity between samples. In addition to a point prediction (posterior mean), the model can return a predictive standard deviation that quantifies uncertainty at each test point.

!!! note
    Training involves inverting the n×n kernel matrix which is **O(n³)**. The model is best suited for datasets up to approximately 5 000 samples. For larger data consider inducing-point approximations or a different regressor.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | lengthScale | 1.0 | float | Length-scale ℓ of the RBF kernel. Larger values produce smoother functions. |
| 2 | amplitude | 1.0 | float | Signal amplitude (output scale) of the RBF kernel. |
| 3 | noiseVariance | 1e-4 | float | Independent Gaussian noise variance σ²_n added to the kernel diagonal. |

## Example
```php
use Rubix\ML\Regressors\GaussianProcessRegressor;

$estimator = new GaussianProcessRegressor(
    lengthScale: 1.5,
    amplitude: 1.0,
    noiseVariance: 1e-3
);
```

## Additional Methods
Return the posterior mean and standard deviation for each sample as an associative array with keys `mean` and `std`:
```php
public predictWithVariance(Dataset $dataset) : array
```

```php
['mean' => $means, 'std' => $stds] = $estimator->predictWithVariance($dataset);

foreach ($means as $i => $mu) {
    echo "Predicted: $mu ± {$stds[$i]}\n";
}
```

## Notes
- The kernel is `k(x, x') = amplitude² × exp(-‖x - x'‖² / (2ℓ²))`.
- Predictions are the posterior mean: `μ* = K(X*, X) × K(X,X)⁻¹ × y`.
- A small `noiseVariance` improves fit but may cause numerical instability on near-duplicate samples.

## References
[^1]: C. E. Rasmussen & C. K. I. Williams. (2006). Gaussian Processes for Machine Learning.
