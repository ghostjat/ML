<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Clusterers/SpectralClustering.php">[source]</a></span>

# Spectral Clustering
Partitions data by embedding samples into the low-dimensional eigenspace of the **normalised graph Laplacian** and then applying K-Means on the resulting embedding. Because the algorithm operates on pairwise similarities rather than raw Euclidean distances, it can discover **non-convex** and **ring-shaped** clusters that K-Means would fail to separate.

The algorithm proceeds as follows:

1. Build RBF affinity matrix: `A_ij = exp(-‖x_i − x_j‖² × γ)`
2. Normalise: `L_sym = D^{-½} A D^{-½}` where `D_ii = Σ_j A_ij`
3. Extract the top-*k* eigenvectors of `L_sym`
4. Row-normalise the eigenvector matrix
5. Apply K-Means++ to the normalised embedding

!!! note
    Building the affinity matrix is **O(n²)**. Recommended for datasets up to ~3 000 samples. For larger datasets, consider a sparse approximation.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | k | 3 | int | Number of clusters (must be ≥ 2). |
| 2 | gamma | 1.0 | float | RBF kernel bandwidth — controls the width of the similarity function. |
| 3 | kMeansEpochs | 300 | int | Maximum Lloyd's K-Means iterations on the spectral embedding. |

## Example
```php
use Rubix\ML\Clusterers\SpectralClustering;

$estimator = new SpectralClustering(
    k: 4,
    gamma: 0.5,
    kMeansEpochs: 500
);
```

## Notes
- `gamma` plays the same role as the bandwidth in a Gaussian kernel: larger values narrow the effective neighbourhood.  A good starting point is `1 / numFeatures`.
- Out-of-sample predictions use an affinity-based Nyström approximation to the spectral embedding.
- The internal K-Means uses K-Means++ seeding for stable initialisation.

## References
[^1]: A. Y. Ng et al. (2002). On Spectral Clustering: Analysis and an algorithm.
