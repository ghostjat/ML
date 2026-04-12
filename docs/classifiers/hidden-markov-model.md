<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Classifiers/HiddenMarkovModel.php">[source]</a></span>

# Hidden Markov Model
A generative sequence classifier that trains one **Gaussian HMM** per class label using the **Baum-Welch** (Forward-Backward EM) algorithm and classifies samples by selecting the model that assigns the highest log-likelihood under the **forward algorithm**. Each sample is treated as a fixed-length observation sequence where every feature dimension is one time step, with diagonal-covariance multivariate Gaussian emissions.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Probabilistic](../probabilistic.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | states | 3 | int | Number of hidden states per class HMM. |
| 2 | epochs | 100 | int | Maximum Baum-Welch EM iterations. |
| 3 | tol | 1e-4 | float | Convergence tolerance on log-likelihood improvement. |
| 4 | minVariance | 1e-6 | float | Variance floor to prevent degenerate Gaussian components. |

## Example
```php
use Rubix\ML\Classifiers\HiddenMarkovModel;

$estimator = new HiddenMarkovModel(
    states: 5,
    epochs: 50,
    tol: 1e-5,
    minVariance: 1e-6
);
```

## Additional Methods
Return the posterior class probability distribution for each sample (softmax of per-class log-likelihoods):
```php
public proba(Dataset $dataset) : array
```

## Notes
- Each feature dimension is treated as an independent time step: a 10-feature sample is a 10-step sequence.
- The transition matrix is initialised with a left-to-right bias (self-loop 0.95, others 0.05).
- Emission means are seeded from evenly-spaced training samples; variances start at 1.
- Forward / backward passes run in log-space to avoid numerical underflow on long sequences.

## References
[^1]: L. R. Rabiner. (1989). A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition.
[^2]: L. E. Baum et al. (1970). A Maximization Technique Occurring in the Statistical Analysis of Probabilistic Functions of Markov Chains.
