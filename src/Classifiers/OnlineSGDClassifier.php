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
use function exp;
use function log;
use function sqrt;
use function max;
use function array_fill;
use function array_keys;
use function array_unique;
use function get_object_vars;
use function is_null;

use const Rubix\ML\EPSILON;

/**
 * Online SGD Classifier
 *
 * A multi-class linear classifier trained incrementally via Stochastic Gradient
 * Descent.  Three loss functions are available: *log* (logistic regression /
 * cross-entropy), *hinge* (linear SVM), and *perceptron*.  Updates use AdaGrad
 * per-feature accumulated gradients for adaptive step-size control, which is
 * well-suited for sparse or large-scale problems.  OvR (one-vs-rest) is used
 * for multi-class extension.
 *
 * References:
 * [1] J. Duchi et al. (2011). Adaptive Subgradient Methods for Online Learning and Stochastic Optimization.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      The Rubix ML Community
 */
class OnlineSGDClassifier implements Estimator, Learner, Online, Probabilistic, Verbose, Persistable
{
    use AutotrackRevisions, LoggerAware;

    /**
     * Supported loss functions.
     */
    protected const LOSSES = ['log', 'hinge', 'perceptron'];

    /**
     * Initial learning rate.
     *
     * @var float
     */
    protected float $rate;

    /**
     * L2 regularization strength.
     *
     * @var float
     */
    protected float $l2Penalty;

    /**
     * The loss function (log | hinge | perceptron).
     *
     * @var string
     */
    protected string $loss;

    /**
     * Number of epochs over the dataset per train() call.
     *
     * @var int
     */
    protected int $epochs;

    /**
     * Number of samples per mini-batch.
     *
     * @var int
     */
    protected int $batchSize;

    /**
     * Per-class weight vectors: class => float[p].
     *
     * @var array<string|int, float[]>|null
     */
    protected ?array $weights = null;

    /**
     * Per-class bias terms: class => float.
     *
     * @var array<string|int, float>|null
     */
    protected ?array $biases = null;

    /**
     * AdaGrad accumulated squared gradients per class per feature.
     *
     * @var array<string|int, float[]>|null
     */
    protected ?array $gCache = null;

    /**
     * AdaGrad accumulated squared gradient for bias per class.
     *
     * @var array<string|int, float>|null
     */
    protected ?array $gBias = null;

    /**
     * Ordered class labels.
     *
     * @var list<string|int>|null
     */
    protected ?array $classes = null;

    /**
     * Number of features seen during initialisation.
     *
     * @var int|null
     */
    protected ?int $featureCount = null;

    /**
     * @param float $rate
     * @param float $l2Penalty
     * @param string $loss
     * @param int $epochs
     * @param int $batchSize
     * @throws InvalidArgumentException
     */
    public function __construct(
        float $rate = 0.01,
        float $l2Penalty = 1e-4,
        string $loss = 'log',
        int $epochs = 100,
        int $batchSize = 64
    ) {
        if ($rate <= 0.0) {
            throw new InvalidArgumentException('Learning rate must be'
                . " greater than 0, $rate given.");
        }

        if ($l2Penalty < 0.0) {
            throw new InvalidArgumentException('L2 penalty must be'
                . " greater than or equal to 0, $l2Penalty given.");
        }

        if (!in_array($loss, self::LOSSES, true)) {
            throw new InvalidArgumentException("Loss must be one of '"
                . implode("', '", self::LOSSES) . "', '$loss' given.");
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Epochs must be greater'
                . " than 0, $epochs given.");
        }

        if ($batchSize < 1) {
            throw new InvalidArgumentException('Batch size must be'
                . " greater than 0, $batchSize given.");
        }

        $this->rate      = $rate;
        $this->l2Penalty = $l2Penalty;
        $this->loss      = $loss;
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
            'rate'       => $this->rate,
            'l2 penalty' => $this->l2Penalty,
            'loss'       => $this->loss,
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
     * Return the weight vectors indexed by class label.
     *
     * @return array<string|int, float[]>|null
     */
    public function weights() : ?array
    {
        return $this->weights;
    }

    /**
     * Train the classifier from scratch.
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
        $classes = array_values(array_unique($labels));

        $weights = [];
        $biases  = [];
        $gCache  = [];
        $gBias   = [];

        foreach ($classes as $class) {
            $weights[$class] = array_fill(0, $p, 0.0);
            $biases[$class]  = 0.0;
            $gCache[$class]  = array_fill(0, $p, 0.0);
            $gBias[$class]   = 0.0;
        }

        $this->weights      = $weights;
        $this->biases       = $biases;
        $this->gCache       = $gCache;
        $this->gBias        = $gBias;
        $this->classes      = $classes;
        $this->featureCount = $p;

        $this->partial($dataset);
    }

    /**
     * Continue training on additional data.
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

        if ($this->logger) {
            $this->logger->info("Training $this");
        }

        /** @var \Rubix\ML\Datasets\Labeled $dataset */
        $samples  = $dataset->samples();
        $labels   = $dataset->labels();
        $n        = count($samples);
        $classes  = $this->classes;
        $nClasses = count($classes);
        $lambda   = $this->l2Penalty;

        for ($epoch = 1; $epoch <= $this->epochs; ++$epoch) {
            $indices = range(0, $n - 1);
            shuffle($indices);

            $bStart = 0;
            while ($bStart < $n) {
                $bEnd = min($bStart + $this->batchSize, $n);

                for ($bi = $bStart; $bi < $bEnd; ++$bi) {
                    $idx       = $indices[$bi];
                    $x         = $samples[$idx];
                    $trueClass = $labels[$idx];
                    $p         = count($x);

                    $this->updateWeights($x, $trueClass, $classes, $p, $lambda);
                }

                $bStart = $bEnd;
            }

            if ($this->logger) {
                $this->logger->info("Epoch: $epoch");
            }
        }

        if ($this->logger) {
            $this->logger->info('Training complete');
        }
    }

    /**
     * Perform one SGD update step for a single sample.
     *
     * @param list<int|float> $x
     * @param string|int $trueClass
     * @param list<string|int> $classes
     * @param int $p
     * @param float $lambda
     */
    protected function updateWeights(array $x, $trueClass, array $classes, int $p, float $lambda) : void
    {
        foreach ($classes as $c) {
            $isPositive = $c === $trueClass;

            $score = $this->biases[$c];
            for ($j = 0; $j < $p; ++$j) {
                $score += $this->weights[$c][$j] * $x[$j];
            }

            // Compute gradient of loss w.r.t. score.
            $grad = $this->lossGradient($score, $isPositive);

            // AdaGrad bias update.
            $this->gBias[$c] += $grad * $grad;
            $aBias = $this->rate / sqrt($this->gBias[$c] + EPSILON);
            $this->biases[$c] -= $aBias * $grad;

            // AdaGrad weight updates.
            for ($j = 0; $j < $p; ++$j) {
                $gj = $grad * $x[$j] + $lambda * $this->weights[$c][$j];
                $this->gCache[$c][$j] += $gj * $gj;
                $aj = $this->rate / sqrt($this->gCache[$c][$j] + EPSILON);
                $this->weights[$c][$j] -= $aj * $gj;
            }
        }
    }

    /**
     * Gradient of the loss w.r.t. the linear score for one OvR binary problem.
     *
     * @param float $score
     * @param bool $isPositive
     * @return float
     */
    protected function lossGradient(float $score, bool $isPositive) : float
    {
        $y = $isPositive ? 1.0 : -1.0;

        switch ($this->loss) {
            case 'log':
                // Logistic: σ(s) - y_01;  rearranged for ±1 labels:
                $sigma = 1.0 / (1.0 + exp(-$score));
                return $sigma - ($isPositive ? 1.0 : 0.0);

            case 'hinge':
                // max(0, 1 - y * s)  →  -y if y*s < 1 else 0
                return ($y * $score < 1.0) ? -$y : 0.0;

            default:  // perceptron
                // -y if mis-classified, else 0
                return ($y * $score <= 0.0) ? -$y : 0.0;
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
                $score = $this->biases[$c];
                foreach ($this->weights[$c] as $j => $wj) {
                    $score += $wj * $x[$j];
                }
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
     * Return softmax class probabilities for each sample.
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
            $scores   = [];
            $maxScore = -INF;

            foreach ($this->classes as $c) {
                $s = $this->biases[$c];
                foreach ($this->weights[$c] as $j => $wj) {
                    $s += $wj * $x[$j];
                }
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
     * @internal
     *
     * @return mixed[]
     */
    public function __serialize() : array
    {
        $properties = get_object_vars($this);

        unset($properties['logger']);

        return $properties;
    }

    /**
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Online SGD Classifier (' . Params::stringify($this->params()) . ')';
    }
}
