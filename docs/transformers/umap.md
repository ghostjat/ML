<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/UMAP.php">[source]</a></span>

# UMAP
*Uniform Manifold Approximation and Projection* is a dimensionality reduction technique that preserves both **local** and **global** structure of the data. Compared to [t-SNE](t-sne.md), UMAP is generally faster, scales better to large datasets, and produces embeddings where inter-cluster distances are more meaningful.

The algorithm models the data as a **fuzzy topological structure** in high-dimensional space, then optimises a low-dimensional representation to match it using stochastic gradient descent with alternating attractive and repulsive forces:

- **Attractive** force pulls connected neighbours together.
- **Repulsive** force pushes randomly sampled non-neighbours apart.

The high-dimensional fuzzy membership between two points is:
```
μ_ij = exp(-max(d_ij − ρ_i, 0) / σ_i)
```
where ρ_i is the distance to the nearest neighbour of point *i* and σ_i is found by binary search.

The low-dimensional similarity uses the curve `q_ij = (1 + a·‖y_i − y_j‖^{2b})^{-1}`, where *a* and *b* are derived from *minDist* and *spread*.

!!! note
    The brute-force k-NN graph construction is **O(n²)**. For datasets larger than ~10 000 samples consider pre-computing neighbours externally.

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful), [Verbose](../verbose.md)

**Data Type Compatibility:** Depends on distance kernel

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | dimensions | 2 | int | Number of output dimensions. |
| 2 | neighbours | 15 | int | Number of nearest neighbours used to build the fuzzy graph. |
| 3 | minDist | 0.1 | float | Minimum distance in the embedding (controls cluster compactness). |
| 4 | epochs | 200 | int | Number of SGD optimisation epochs. |
| 5 | rate | 1.0 | float | Initial learning rate (linearly decays to 0). |
| 6 | spread | 1.0 | float | Effective scale of the embedding (must be > *minDist*). |
| 7 | kernel | Euclidean | Distance | Distance kernel for computing pairwise distances. |

## Example
```php
use Rubix\ML\Transformers\UMAP;
use Rubix\ML\Kernels\Distance\Euclidean;

$transformer = new UMAP(
    dimensions: 2,
    neighbours: 15,
    minDist: 0.1,
    epochs: 200,
    rate: 1.0,
    spread: 1.0,
    kernel: new Euclidean()
);

// Fit on training data then transform:
$transformer->fit($dataset);
$dataset->apply($transformer);
```

## Usage Pattern
UMAP implements the `Stateful` interface, so you must call `fit()` before `transform()`.  `transform()` on the same dataset that was fitted returns the computed embedding directly; on new samples it uses a Nadaraya-Watson kernel regression over the training embedding for out-of-sample projection.

```php
// Check if fitted:
$transformer->fitted(); // bool

// Fit on training data:
$transformer->fit($trainDataset);

// Apply to training data (returns stored embedding):
$trainDataset->apply($transformer);

// Project new samples:
$testDataset->apply($transformer);
```

## Notes
- Increase `neighbours` (e.g. 30–50) to better preserve global structure at the cost of local detail.
- Decrease `minDist` (e.g. 0.01) for tighter, more separated clusters; increase it (e.g. 0.5) for a more uniform spread.
- The learning rate decays linearly to 0 over `epochs` iterations.

## References
[^1]: L. McInnes et al. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.
