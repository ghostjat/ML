<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Learner;
use Rubix\ML\Online;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
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
use function exp;
use function log;
use function sqrt;
use function array_fill;
use function array_keys;
use function is_null;

use const Rubix\ML\EPSILON;

/**
 * Factorization Machine Classifier
 *
 * A second-order feature interaction model trained with mini-batch SGD using the
 * logistic (cross-entropy) loss.  The interaction term is computed in O(kn) time
 * using the identity: Σ_{i<j} <v_i,v_j> x_i x_j = 0.5 Σ_f [(Σ_i v_{if} x_i)² - Σ_i v²_{if} x²_i].
 * Supports binary and multi-class problems via a per-class set of FM parameters.
 *
 * References:
 * [1] S. Rendle. (2010). Factorization Machines.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      The Rubix ML Community
 */
class FactorizationMachine implements Estimator, Learner, Online, Probabilistic, Persistable
{
    use AutotrackRevisions;

    /**
     * Number of latent factors per feature.
     *
     * @var int
     */
    protected int $factors;

    /**
     * Initial learning rate for SGD.
     *
     * @var float
     */
    protected float $rate;

    /**
     * L2 regularization on weights and factor vectors.
     *
     * @var float
     */
    protected float $l2Penalty;

    /**
     * Number of training epochs.
     *
     * @var int
     */
    protected int $epochs;

    /**
     * Samples per mini-batch.
     *
     * @var int
     */
    protected int $batchSize;

    /**
     * Global bias per class: class => float.
     *
     * @var float[]|null
     */
    protected ?array $biases = null;

    /**
     * First-order weights per class: class => float[p].
     *
     * @var array<string|int, float[]>|null
     */
    protected ?array $weights = null;

    /**
     * Factor matrices per class: class => float[p][k].
     *
     * @var array<string|int, float[][]>|null
     */
    protected ?array $factors2 = null;

    /**
     * Ordered class labels.
     *
     * @var list<string|int>|null
     */
    protected ?array $classes = null;

    /**
     * Number of input features seen during training.
     *
     * @var int|null
     */
    protected ?int $featureCount = null;

    /**
     * @param int $factors
     * @param float $rate
     * @param float $l2Penalty
     * @param int $epochs
     * @param int $batchSize
     * @throws InvalidArgumentException
     */
    public function __construct(
        int $factors = 10,
        float $rate = 0.01,
        float $l2Penalty = 1e-4,
        int $epochs = 100,
        int $batchSize = 64
    ) {
        if ($factors < 1) {
            throw new InvalidArgumentException('Factors must be greater'
                . " than 0, $factors given.");
        }

        if ($rate <= 0.0) {
            throw new InvalidArgumentException('Learning rate must be'
                . " greater than 0, $rate given.");
        }

        if ($l2Penalty < 0.0) {
            throw new InvalidArgumentException('L2 penalty must be'
                . " greater than or equal to 0, $l2Penalty given.");
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Epochs must be greater'
                . " than 0, $epochs given.");
        }

        if ($batchSize < 1) {
            throw new InvalidArgumentException('Batch size must be'
                . " greater than 0, $batchSize given.");
        }

        $this->factors   = $factors;
        $this->rate      = $rate;
        $this->l2Penalty = $l2Penalty;
        $this->epochs    = $epochs;
        $this->batchSize = $batchSize;
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
            'factors'    => $this->factors,
            'rate'       => $this->rate,
            'l2 penalty' => $this->l2Penalty,
            'epochs'     => $this->epochs,
            'batch size' => $this->batchSize,
        ];
    }

    /**
     * @return bool
     */
    public function trained() : bool
    {
        return isset($this->weights);
    }

    /**
     * Train on a labeled dataset from scratch.
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
        $labels  = $dataset->labels();
        $p       = $dataset->numFeatures();
        $k       = $this->factors;
        $classes = array_unique($labels);

        $scale = 1.0 / sqrt($k);

        $biases  = [];
        $weights = [];
        $factors2 = [];

        foreach ($classes as $class) {
            $biases[$class]  = 0.0;
            $w = array_fill(0, $p, 0.0);
            $weights[$class] = $w;

            // Random init: N(0, 1/sqrt(k))
            $V = [];
            for ($j = 0; $j < $p; ++$j) {
                $row = [];
                for ($f = 0; $f < $k; ++$f) {
                    $row[] = (lcg_value() * 2.0 - 1.0) * $scale;
                }
                $V[] = $row;
            }
            $factors2[$class] = $V;
        }

        $this->biases      = $biases;
        $this->weights     = $weights;
        $this->factors2    = $factors2;
        $this->classes     = array_values($classes);
        $this->featureCount = $p;

        $this->partial($dataset);
    }

    /**
     * Perform an incremental training pass.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     */
    public function partial(Dataset $dataset) : void
    {
        if (!isset($this->weights)) {
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
        $samples  = $dataset->samples();
        $labels   = $dataset->labels();
        $n        = count($samples);
        $k        = $this->factors;
        $classes  = $this->classes;
        $nClasses = count($classes);
        $alpha    = $this->rate;
        $lambda   = $this->l2Penalty;

        for ($epoch = 0; $epoch < $this->epochs; ++$epoch) {
            // Shuffle indices for mini-batch SGD.
            $indices = range(0, $n - 1);
            shuffle($indices);

            $bStart = 0;
            while ($bStart < $n) {
                $bEnd = min($bStart + $this->batchSize, $n);

                for ($bi = $bStart; $bi < $bEnd; ++$bi) {
                    $idx    = $indices[$bi];
                    $x      = $samples[$idx];
                    $trueClass = $labels[$idx];

                    // Compute scores for all classes.
                    $scores = [];
                    foreach ($classes as $c) {
                        $scores[$c] = $this->fmScore($x, $c);
                    }

                    // Softmax.
                    $maxScore = max($scores);
                    $expScores = [];
                    $expSum    = 0.0;
                    foreach ($scores as $c => $s) {
                        $e            = exp($s - $maxScore);
                        $expScores[$c] = $e;
                        $expSum       += $e;
                    }

                    // Gradient and update for each class (OvR via softmax).
                    foreach ($classes as $c) {
                        $prob  = $expScores[$c] / ($expSum ?: EPSILON);
                        $delta = $prob - ($c === $trueClass ? 1.0 : 0.0);

                        $p = count($x);

                        // Update bias.
                        $this->biases[$c] -= $alpha * $delta;

                        // Precompute Σ_i v_{if} x_i for all factors.
                        $V       = $this->factors2[$c];
                        $sumVX   = array_fill(0, $k, 0.0);
                        for ($j = 0; $j < $p; ++$j) {
                            $xj = $x[$j];
                            for ($f = 0; $f < $k; ++$f) {
                                $sumVX[$f] += $V[$j][$f] * $xj;
                            }
                        }

                        for ($j = 0; $j < $p; ++$j) {
                            $xj = $x[$j];

                            // dL/dw_j
                            $this->weights[$c][$j] -= $alpha * ($delta * $xj + $lambda * $this->weights[$c][$j]);

                            // dL/dV_{jf}
                            for ($f = 0; $f < $k; ++$f) {
                                $grad = $delta * $xj * ($sumVX[$f] - $V[$j][$f] * $xj);
                                $this->factors2[$c][$j][$f] -= $alpha * ($grad + $lambda * $V[$j][$f]);
                            }
                        }
                    }
                }

                $bStart = $bEnd;
            }
        }
    }

    /**
     * Predict the most probable class for each sample.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<string|int>
     */
    public function predict(Dataset $dataset) : array
    {
        if (!isset($this->classes)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->featureCount)->check();

        $predictions = [];

        foreach ($dataset->samples() as $x) {
            $bestClass = null;
            $bestScore = -INF;

            foreach ($this->classes as $c) {
                $score = $this->fmScore($x, $c);
                if ($score > $bestScore) {
                    $bestScore = $score;
                    $bestClass = $c;
                }
            }

            $predictions[] = $bestClass;
        }

        return $predictions;
    }

    /**
     * Return the softmax class probabilities for each sample.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<float[]>
     */
    public function proba(Dataset $dataset) : array
    {
        if (!isset($this->classes)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->featureCount)->check();

        $probas = [];

        foreach ($dataset->samples() as $x) {
            $scores  = [];
            $maxScore = -INF;

            foreach ($this->classes as $c) {
                $s = $this->fmScore($x, $c);
                $scores[$c] = $s;
                if ($s > $maxScore) {
                    $maxScore = $s;
                }
            }

            $expSum = 0.0;
            $exps   = [];
            foreach ($scores as $c => $s) {
                $e          = exp($s - $maxScore);
                $exps[$c]   = $e;
                $expSum    += $e;
            }

            $dist = [];
            foreach ($exps as $c => $e) {
                $dist[$c] = $e / ($expSum ?: EPSILON);
            }

            $probas[] = $dist;
        }

        return $probas;
    }

    /**
     * Compute FM score for a single sample and class.
     *
     * y = w0 + Σ w_j x_j + 0.5 Σ_f [(Σ_j v_{jf} x_j)² - Σ_j v²_{jf} x²_j]
     *
     * @param list<int|float> $x
     * @param string|int $class
     * @return float
     */
    protected function fmScore(array $x, $class) : float
    {
        $score = $this->biases[$class];
        $w     = $this->weights[$class];
        $V     = $this->factors2[$class];
        $k     = $this->factors;
        $p     = count($x);

        // First-order terms.
        for ($j = 0; $j < $p; ++$j) {
            $score += $w[$j] * $x[$j];
        }

        // Second-order interaction terms in O(kp).
        $inter = 0.0;
        for ($f = 0; $f < $k; ++$f) {
            $sumVX  = 0.0;
            $sumVX2 = 0.0;
            for ($j = 0; $j < $p; ++$j) {
                $vxj    = $V[$j][$f] * $x[$j];
                $sumVX += $vxj;
                $sumVX2 += $vxj * $vxj;
            }
            $inter += $sumVX * $sumVX - $sumVX2;
        }

        return $score + 0.5 * $inter;
    }

    /**
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Factorization Machine (Classifier) (' . Params::stringify($this->params()) . ')';
    }
}
