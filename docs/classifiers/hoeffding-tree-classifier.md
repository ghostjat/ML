<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Classifiers/HoeffdingTreeClassifier.php">[source]</a></span>

# Hoeffding Tree Classifier
A *Very Fast Decision Tree* (VFDT) that learns incrementally from a stream of labeled samples **without storing the full dataset**. Each leaf accumulates per-class, per-feature sufficient statistics (count, sum, sum-of-squares). When a leaf has seen at least *minSamples* examples, a split is attempted using the **Hoeffding bound** to guarantee — with confidence 1 − δ — that the best split attribute is chosen correctly.

The Hoeffding bound states:

```
ε = √( R² × ln(1/δ) / (2n) )
```

A split is triggered when the information gain gap between the best and second-best attribute exceeds ε, or when ε < τ (tie-breaking).

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Online](../online.md), [Probabilistic](../probabilistic.md), [Verbose](../verbose.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | delta | 1e-7 | float | Confidence parameter δ for the Hoeffding bound (0 < δ < 1). |
| 2 | tau | 0.05 | float | Tie-breaking threshold τ; split when ε < τ even if the gap is small. |
| 3 | minSamples | 200 | int | Minimum samples at a leaf before attempting a split. |
| 4 | maxDepth | 0 | int | Maximum tree depth (0 = unlimited). |
| 5 | minGain | 1e-4 | float | Minimum information gain required for a split to be accepted. |

## Example
```php
use Rubix\ML\Classifiers\HoeffdingTreeClassifier;

$estimator = new HoeffdingTreeClassifier(
    delta: 1e-7,
    tau: 0.05,
    minSamples: 200,
    maxDepth: 20,
    minGain: 1e-4
);

// Stream training — safe to call partial() millions of times:
foreach ($streamBatches as $batch) {
    $estimator->partial($batch);
}
```

## Additional Methods
Return the class probability distribution at the reaching leaf for each sample:
```php
public proba(Dataset $dataset) : array
```

## Notes
- Split statistics are approximated using per-class Gaussian CDFs, avoiding the need to buffer raw samples.
- The tree is stored as a flat PHP array of node descriptors — no recursion in hot prediction paths.
- Lower `delta` = more conservative splits (larger trees, more accurate); higher `delta` = faster splits.
- Set `maxDepth` to prevent runaway growth on noisy streams.

## References
[^1]: P. Domingos & G. Hulten. (2000). Mining High-Speed Data Streams.
