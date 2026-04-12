<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Online;
use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Traits\LoggerAware;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\DatasetIsLabeled;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function count;
use function log;
use function sqrt;
use function array_fill;
use function array_keys;
use function array_sum;
use function array_unique;
use function max;

use const Rubix\ML\EPSILON;
use const M_LN2;

/**
 * Hoeffding Tree Classifier
 *
 * An anytime decision-tree classifier (VFDT — Very Fast Decision Tree) that learns
 * from a stream of labeled samples without storing the full dataset.  Each leaf
 * maintains sufficient statistics (per-class feature sums and squared sums) to
 * evaluate candidate splits on demand.  A split is triggered when the Hoeffding
 * bound guarantees that the best attribute is statistically better than the
 * second-best by more than the bound ε, or when the two are too close to distinguish
 * (tie-breaking).
 *
 * References:
 * [1] P. Domingos & G. Hulten. (2000). Mining High-Speed Data Streams.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      The Rubix ML Community
 */
class HoeffdingTreeClassifier implements Estimator, Learner, Online, Probabilistic, Verbose, Persistable
{
    use AutotrackRevisions, LoggerAware;

    /**
     * Confidence parameter δ for the Hoeffding bound (lower = more conservative).
     *
     * @var float
     */
    protected float $delta;

    /**
     * Tie-breaking threshold τ: split anyway when top-two gain difference < τ.
     *
     * @var float
     */
    protected float $tau;

    /**
     * Minimum samples at a leaf before a split is attempted.
     *
     * @var int
     */
    protected int $minSamples;

    /**
     * Maximum tree depth (0 = unlimited).
     *
     * @var int
     */
    protected int $maxDepth;

    /**
     * Minimum information gain required for a split to be accepted.
     *
     * @var float
     */
    protected float $minGain;

    /**
     * Flat node store indexed by integer node id.
     * Each node is one of:
     *   Internal: ['split', feature_idx, threshold, left_id, right_id, depth]
     *   Leaf:     ['leaf', n, class_counts, feat_class_stats, depth]
     *
     * feat_class_stats[j][c] = [count, sum, sqSum]
     *
     * @var array<int, mixed[]>
     */
    protected array $nodes = [];

    /**
     * Auto-incrementing node-id counter.
     *
     * @var int
     */
    protected int $nextId = 0;

    /**
     * Root node id.
     *
     * @var int|null
     */
    protected ?int $rootId = null;

    /**
     * Total number of classes seen.
     *
     * @var int
     */
    protected int $nClasses = 0;

    /**
     * Number of input features.
     *
     * @var int|null
     */
    protected ?int $featureCount = null;

    /**
     * @param float $delta
     * @param float $tau
     * @param int $minSamples
     * @param int $maxDepth
     * @param float $minGain
     * @throws InvalidArgumentException
     */
    public function __construct(
        float $delta = 1e-7,
        float $tau = 0.05,
        int $minSamples = 200,
        int $maxDepth = 0,
        float $minGain = 1e-4
    ) {
        if ($delta <= 0.0 or $delta >= 1.0) {
            throw new InvalidArgumentException('Delta must be between'
                . " 0 and 1, $delta given.");
        }

        if ($tau < 0.0) {
            throw new InvalidArgumentException('Tau must be greater'
                . " than or equal to 0, $tau given.");
        }

        if ($minSamples < 1) {
            throw new InvalidArgumentException('Min samples must be'
                . " greater than 0, $minSamples given.");
        }

        if ($maxDepth < 0) {
            throw new InvalidArgumentException('Max depth must be'
                . " greater than or equal to 0, $maxDepth given.");
        }

        if ($minGain < 0.0) {
            throw new InvalidArgumentException('Min gain must be'
                . " greater than or equal to 0, $minGain given.");
        }

        $this->delta      = $delta;
        $this->tau        = $tau;
        $this->minSamples = $minSamples;
        $this->maxDepth   = $maxDepth;
        $this->minGain    = $minGain;
    }

    /**
     * @internal
     */
    public function type() : EstimatorType
    {
        return EstimatorType::classifier();
    }

    /**
     * @internal
     *
     * @return list<DataType>
     */
    public function compatibility() : array
    {
        return [
            DataType::continuous(),
        ];
    }

    /**
     * @internal
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'delta'       => $this->delta,
            'tau'         => $this->tau,
            'min samples' => $this->minSamples,
            'max depth'   => $this->maxDepth,
            'min gain'    => $this->minGain,
        ];
    }

    /**
     * @return bool
     */
    public function trained() : bool
    {
        return $this->rootId !== null;
    }

    /**
     * Train from scratch on a labeled dataset.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     */
    public function train(Dataset $dataset) : void
    {
        SpecificationChain::with([
            new DatasetIsLabeled($dataset),
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
            new LabelsAreCompatibleWithLearner($dataset, $this),
        ])->check();

        /** @var \Rubix\ML\Datasets\Labeled $dataset */
        $p = $dataset->numFeatures();
        $labels = $dataset->labels();
        $classes = array_values(array_unique($labels));

        $this->nodes       = [];
        $this->nextId      = 0;
        $this->nClasses    = count($classes);
        $this->featureCount = $p;
        $this->rootId      = $this->makeLeaf(0);

        $this->partial($dataset);
    }

    /**
     * Incrementally train on additional data.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     */
    public function partial(Dataset $dataset) : void
    {
        if ($this->rootId === null) {
            $this->train($dataset);
            return;
        }

        SpecificationChain::with([
            new DatasetIsLabeled($dataset),
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
            new LabelsAreCompatibleWithLearner($dataset, $this),
            new DatasetHasDimensionality($dataset, $this->featureCount),
        ])->check();

        /** @var \Rubix\ML\Datasets\Labeled $dataset */
        $samples = $dataset->samples();
        $labels  = $dataset->labels();
        $n       = count($samples);

        for ($i = 0; $i < $n; ++$i) {
            $this->updateLeaf($samples[$i], $labels[$i]);
        }
    }

    /**
     * Predict the majority class at the leaf reached by each sample.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<string|int>
     */
    public function predict(Dataset $dataset) : array
    {
        if ($this->rootId === null) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->featureCount)->check();

        $predictions = [];

        foreach ($dataset->samples() as $sample) {
            $leaf = $this->traverse($sample);
            $predictions[] = $this->majorityClass($leaf);
        }

        return $predictions;
    }

    /**
     * Return the class probability distribution at the leaf for each sample.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<float[]>
     */
    public function proba(Dataset $dataset) : array
    {
        if ($this->rootId === null) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->featureCount)->check();

        $probas = [];

        foreach ($dataset->samples() as $sample) {
            $leaf = $this->traverse($sample);
            $probas[] = $this->leafProbabilities($leaf);
        }

        return $probas;
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /**
     * Create a new leaf node and return its id.
     *
     * @param int $depth
     * @return int
     */
    protected function makeLeaf(int $depth) : int
    {
        $id = $this->nextId++;
        $this->nodes[$id] = [
            'type'    => 'leaf',
            'n'       => 0,
            'classes' => [],   // class => count
            'stats'   => [],   // [j][class] => [count, sum, sqSum]
            'depth'   => $depth,
        ];
        return $id;
    }

    /**
     * Route a sample down the tree and update the leaf's sufficient statistics.
     *
     * @param list<int|float> $sample
     * @param string|int $label
     */
    protected function updateLeaf(array $sample, $label) : void
    {
        $nodeId = $this->rootId;

        while (true) {
            $node = &$this->nodes[$nodeId];

            if ($node['type'] === 'split') {
                $nodeId = ($sample[$node['feature']] <= $node['threshold'])
                    ? $node['left']
                    : $node['right'];
                continue;
            }

            // Leaf: update statistics.
            $node['n']++;
            $node['classes'][$label] = ($node['classes'][$label] ?? 0) + 1;

            $p = count($sample);
            for ($j = 0; $j < $p; ++$j) {
                $v = (float) $sample[$j];
                if (!isset($node['stats'][$j][$label])) {
                    $node['stats'][$j][$label] = [0, 0.0, 0.0];
                }
                $node['stats'][$j][$label][0]++;
                $node['stats'][$j][$label][1] += $v;
                $node['stats'][$j][$label][2] += $v * $v;
            }

            // Attempt split check.
            if ($node['n'] >= $this->minSamples) {
                $this->attemptSplit($nodeId);
            }

            break;
        }
    }

    /**
     * Evaluate the Hoeffding bound and split the leaf if warranted.
     *
     * @param int $nodeId
     */
    protected function attemptSplit(int $nodeId) : void
    {
        $node = &$this->nodes[$nodeId];

        if ($node['type'] !== 'leaf') {
            return;
        }

        $depth = $node['depth'];

        if ($this->maxDepth > 0 and $depth >= $this->maxDepth) {
            return;
        }

        $nClasses = max($this->nClasses, count($node['classes']));
        if ($nClasses < 2) {
            return;
        }

        // Range of entropy: log2(nClasses) bits.
        $R = log($nClasses) / M_LN2;

        $eps = sqrt($R * $R * log(1.0 / $this->delta) / (2.0 * $node['n']));

        // Compute best split per feature.
        $gains = [];
        $thresholds = [];

        $baseEntropy = $this->entropy($node['classes'], $node['n']);

        $p = count($node['stats']);

        for ($j = 0; $j < $p; ++$j) {
            $featStats = $node['stats'][$j] ?? [];

            if (empty($featStats)) {
                $gains[$j]      = 0.0;
                $thresholds[$j] = 0.0;
                continue;
            }

            // Candidate threshold: mean of per-class feature means.
            $meanSum = 0.0;
            $totalCount = 0;
            foreach ($featStats as $c => [$cnt, $sum, $sqSum]) {
                $meanSum    += $cnt > 0 ? $sum / $cnt : 0.0;
                $totalCount += $cnt;
            }
            $threshold = $meanSum / max(count($featStats), 1);

            // Estimate class distributions on each side using Gaussian CDF approx.
            [$leftCounts, $rightCounts] = $this->splitCounts($featStats, $threshold, $node['n']);

            $leftN  = array_sum($leftCounts);
            $rightN = array_sum($rightCounts);

            $gain = $baseEntropy;

            if ($leftN > 0) {
                $gain -= ($leftN / $node['n']) * $this->entropy($leftCounts, $leftN);
            }
            if ($rightN > 0) {
                $gain -= ($rightN / $node['n']) * $this->entropy($rightCounts, $rightN);
            }

            $gains[$j]      = $gain;
            $thresholds[$j] = $threshold;
        }

        // Sort features by gain descending.
        arsort($gains);
        $featList = array_keys($gains);

        if (count($featList) < 1) {
            return;
        }

        $bestJ     = $featList[0];
        $bestGain  = $gains[$bestJ];
        $secondJ   = $featList[1] ?? null;
        $secondGain = $secondJ !== null ? $gains[$secondJ] : 0.0;

        if ($bestGain < $this->minGain) {
            return;
        }

        $diff = $bestGain - $secondGain;

        // Hoeffding condition: split if gap > ε, or tie-break if ε < τ.
        if ($diff > $eps or $eps < $this->tau) {
            $this->splitLeaf($nodeId, $bestJ, $thresholds[$bestJ], $depth);
        }
    }

    /**
     * Convert a leaf into an internal split node and create two children.
     *
     * @param int $nodeId
     * @param int $feature
     * @param float $threshold
     * @param int $depth
     */
    protected function splitLeaf(int $nodeId, int $feature, float $threshold, int $depth) : void
    {
        $leftId  = $this->makeLeaf($depth + 1);
        $rightId = $this->makeLeaf($depth + 1);

        if ($this->logger) {
            $this->logger->info("Split node $nodeId on feature $feature @ $threshold");
        }

        $this->nodes[$nodeId] = [
            'type'      => 'split',
            'feature'   => $feature,
            'threshold' => $threshold,
            'left'      => $leftId,
            'right'     => $rightId,
            'depth'     => $depth,
        ];
    }

    /**
     * Traverse the tree for a sample and return the leaf node array (by reference copy).
     *
     * @param list<int|float> $sample
     * @return mixed[]
     */
    protected function traverse(array $sample) : array
    {
        $nodeId = $this->rootId;

        while (true) {
            $node = $this->nodes[$nodeId];

            if ($node['type'] === 'leaf') {
                return $node;
            }

            $nodeId = ($sample[$node['feature']] <= $node['threshold'])
                ? $node['left']
                : $node['right'];
        }
    }

    /**
     * Return the majority class at a leaf node.
     *
     * @param mixed[] $leaf
     * @return string|int
     */
    protected function majorityClass(array $leaf)
    {
        $classes = $leaf['classes'];

        if (empty($classes)) {
            return 0;
        }

        $best     = null;
        $bestCount = -1;
        foreach ($classes as $class => $count) {
            if ($count > $bestCount) {
                $bestCount = $count;
                $best      = $class;
            }
        }

        return $best;
    }

    /**
     * Return a normalised class probability distribution from a leaf.
     *
     * @param mixed[] $leaf
     * @return float[]
     */
    protected function leafProbabilities(array $leaf) : array
    {
        $classes = $leaf['classes'];
        $n       = $leaf['n'] ?: 1;
        $probs   = [];

        foreach ($classes as $class => $count) {
            $probs[$class] = $count / $n;
        }

        return $probs;
    }

    /**
     * Compute Shannon entropy (in bits) of a class-count distribution.
     *
     * @param array<string|int, int|float> $counts
     * @param float $total
     * @return float
     */
    protected function entropy(array $counts, float $total) : float
    {
        if ($total <= 0.0) {
            return 0.0;
        }

        $entropy = 0.0;

        foreach ($counts as $count) {
            if ($count > 0) {
                $p        = $count / $total;
                $entropy -= $p * (log($p) / M_LN2);
            }
        }

        return $entropy;
    }

    /**
     * Approximate the class counts on each side of a split threshold using the
     * per-class Gaussian CDF (logistic approximation Φ(x) ≈ σ(1.7x)).
     *
     * @param array<string|int, array{int, float, float}> $featStats  [class => [count, sum, sqSum]]
     * @param float $threshold
     * @param int $total
     * @return array{array<string|int, float>, array<string|int, float>}
     */
    protected function splitCounts(array $featStats, float $threshold, int $total) : array
    {
        $left  = [];
        $right = [];

        foreach ($featStats as $class => [$cnt, $sum, $sqSum]) {
            if ($cnt < 1) {
                $left[$class]  = 0.0;
                $right[$class] = 0.0;
                continue;
            }

            $mean = $sum / $cnt;
            $var  = max($sqSum / $cnt - $mean * $mean, EPSILON);
            $std  = sqrt($var);

            // Logistic approximation of Gaussian CDF: Φ((t - μ) / σ) ≈ σ(1.7*(t-μ)/σ)
            $z    = 1.7 * ($threshold - $mean) / $std;
            $pLeft = 1.0 / (1.0 + exp(-$z));  // fraction expected to the left

            $left[$class]  = $pLeft * $cnt;
            $right[$class] = (1.0 - $pLeft) * $cnt;
        }

        return [$left, $right];
    }

    /**
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Hoeffding Tree Classifier (' . Params::stringify($this->params()) . ')';
    }
}
