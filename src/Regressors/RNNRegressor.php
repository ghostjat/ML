<?php

declare(strict_types=1);

namespace Rubix\ML\Regressors;

use Rubix\ML\Online;
use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\DataType;
use Rubix\ML\Encoding;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Traits\LoggerAware;
use Rubix\ML\NeuralNet\Snapshot;
use Rubix\ML\NeuralNet\FeedForward;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\NeuralNet\Layers\Recurrent;
use Rubix\ML\NeuralNet\Layers\BlasRecurrent;
use Rubix\ML\Blas\BLASFFI;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Layers\Continuous;
use Rubix\ML\NeuralNet\Layers\Placeholder1D;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Initializers\Xavier2;
use Rubix\ML\CrossValidation\Metrics\RMSE;
use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\NeuralNet\CostFunctions\LeastSquares;
use Rubix\ML\NeuralNet\CostFunctions\RegressionLoss;
use Rubix\ML\Specifications\DatasetIsLabeled;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use Rubix\ML\Specifications\EstimatorIsCompatibleWithMetric;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Generator;

use function is_nan;
use function count;
use function number_format;

/**
 * RNN Regressor
 *
 * An Elman (simple) Recurrent Neural Network for time-series regression. The
 * network processes input sequences step-by-step through a `Recurrent` hidden
 * layer and maps the final hidden state to a scalar output via a `Dense`
 * projection and a `Continuous` output layer.
 *
 * **Dataset format**: each sample must be a flat feature vector of length
 * (sequenceLength × inputSize) where values are laid out as:
 *   [step0_feat0, …, step0_featI−1, step1_feat0, …]
 *
 * Gradients are propagated through all time steps (full BPTT) with
 * element-wise gradient clipping to mitigate exploding gradients.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      The Rubix ML Community
 */
class RNNRegressor implements Estimator, Learner, Online, Verbose, Persistable
{
    use AutotrackRevisions, LoggerAware;

    /**
     * Number of input time steps.
     *
     * @var positive-int
     */
    protected int $sequenceLength;

    /**
     * Features per time step.
     *
     * @var positive-int
     */
    protected int $inputSize;

    /**
     * Number of hidden units in the recurrent layer.
     *
     * @var positive-int
     */
    protected int $hiddenSize;

    /**
     * Extra Dense hidden layers placed between the Recurrent layer and output.
     *
     * @var Hidden[]
     */
    protected array $extraHidden;

    /**
     * Mini-batch size.
     *
     * @var positive-int
     */
    protected int $batchSize;

    /**
     * @var Optimizer
     */
    protected Optimizer $optimizer;

    /**
     * @var float
     */
    protected float $l2Penalty;

    /**
     * @var int<0,max>
     */
    protected int $epochs;

    /**
     * @var float
     */
    protected float $minChange;

    /**
     * @var positive-int
     */
    protected int $window;

    /**
     * @var float
     */
    protected float $holdOut;

    /**
     * @var RegressionLoss
     */
    protected RegressionLoss $costFn;

    /**
     * @var Metric
     */
    protected Metric $metric;

    /**
     * @var FeedForward|null
     */
    protected ?FeedForward $network = null;

    /**
     * @var float[]|null
     */
    protected ?array $scores = null;

    /**
     * @var float[]|null
     */
    protected ?array $losses = null;

    /**
     * @param int              $sequenceLength  Time steps per sample
     * @param int              $inputSize       Features per time step
     * @param int              $hiddenSize      RNN hidden units
     * @param Hidden[]         $extraHidden     Additional Dense layers after RNN
     * @param int              $batchSize
     * @param Optimizer|null   $optimizer
     * @param float            $l2Penalty
     * @param int              $epochs
     * @param float            $minChange
     * @param int              $window
     * @param float            $holdOut
     * @param RegressionLoss|null $costFn
     * @param Metric|null      $metric
     * @throws InvalidArgumentException
     */
    public function __construct(
        int $sequenceLength = 24,
        int $inputSize = 1,
        int $hiddenSize = 64,
        array $extraHidden = [],
        int $batchSize = 32,
        ?Optimizer $optimizer = null,
        float $l2Penalty = 1e-4,
        int $epochs = 500,
        float $minChange = 1e-4,
        int $window = 5,
        float $holdOut = 0.1,
        ?RegressionLoss $costFn = null,
        ?Metric $metric = null
    ) {
        if ($sequenceLength < 1) {
            throw new InvalidArgumentException("Sequence length must be"
                . " at least 1, $sequenceLength given.");
        }

        if ($inputSize < 1) {
            throw new InvalidArgumentException("Input size must be"
                . " at least 1, $inputSize given.");
        }

        if ($hiddenSize < 1) {
            throw new InvalidArgumentException("Hidden size must be"
                . " at least 1, $hiddenSize given.");
        }

        if ($batchSize < 1) {
            throw new InvalidArgumentException("Batch size must be"
                . " at least 1, $batchSize given.");
        }

        if ($l2Penalty < 0.0) {
            throw new InvalidArgumentException("L2 penalty must be"
                . " non-negative, $l2Penalty given.");
        }

        if ($epochs < 0) {
            throw new InvalidArgumentException("Epochs must be"
                . " non-negative, $epochs given.");
        }

        if ($minChange < 0.0) {
            throw new InvalidArgumentException("Min change must be"
                . " non-negative, $minChange given.");
        }

        if ($window < 1) {
            throw new InvalidArgumentException("Window must be"
                . " at least 1, $window given.");
        }

        if ($holdOut < 0.0 || $holdOut > 0.5) {
            throw new InvalidArgumentException("Hold-out must be"
                . " between 0 and 0.5, $holdOut given.");
        }

        if ($metric) {
            EstimatorIsCompatibleWithMetric::with($this, $metric)->check();
        }

        $this->sequenceLength = $sequenceLength;
        $this->inputSize      = $inputSize;
        $this->hiddenSize     = $hiddenSize;
        $this->extraHidden    = $extraHidden;
        $this->batchSize      = $batchSize;
        $this->optimizer      = $optimizer ?? new Adam();
        $this->l2Penalty      = $l2Penalty;
        $this->epochs         = $epochs;
        $this->minChange      = $minChange;
        $this->window         = $window;
        $this->holdOut        = $holdOut;
        $this->costFn         = $costFn ?? new LeastSquares();
        $this->metric         = $metric ?? new RMSE();
    }

    /**
     * @return EstimatorType
     */
    public function type() : EstimatorType
    {
        return EstimatorType::regressor();
    }

    /**
     * @return list<DataType>
     */
    public function compatibility() : array
    {
        return [DataType::continuous()];
    }

    /**
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'sequence length' => $this->sequenceLength,
            'input size'      => $this->inputSize,
            'hidden size'     => $this->hiddenSize,
            'extra hidden'    => $this->extraHidden,
            'batch size'      => $this->batchSize,
            'optimizer'       => $this->optimizer,
            'l2 penalty'      => $this->l2Penalty,
            'epochs'          => $this->epochs,
            'min change'      => $this->minChange,
            'window'          => $this->window,
            'hold out'        => $this->holdOut,
            'cost fn'         => $this->costFn,
            'metric'          => $this->metric,
        ];
    }

    /**
     * @return bool
     */
    public function trained() : bool
    {
        return isset($this->network);
    }

    /**
     * @return Generator<mixed[]>
     */
    public function steps() : Generator
    {
        if (!$this->losses) {
            return;
        }

        foreach ($this->losses as $epoch => $loss) {
            yield [
                'epoch' => $epoch,
                'score' => $this->scores[$epoch] ?? null,
                'loss'  => $loss,
            ];
        }
    }

    /**
     * @return float[]|null
     */
    public function scores() : ?array
    {
        return $this->scores;
    }

    /**
     * @return float[]|null
     */
    public function losses() : ?array
    {
        return $this->losses;
    }

    /**
     * @return FeedForward|null
     */
    public function network() : ?FeedForward
    {
        return $this->network;
    }

    /**
     * Train the RNN on a labeled dataset.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     */
    public function train(Dataset $dataset) : void
    {
        DatasetIsNotEmpty::with($dataset)->check();

        $numInputs = $this->sequenceLength * $this->inputSize;

        $hidden = [
            BLASFFI::isAvailable()
                ? new BlasRecurrent($this->hiddenSize, $this->inputSize)
                : new Recurrent($this->hiddenSize, $this->inputSize),
        ];

        foreach ($this->extraHidden as $layer) {
            $hidden[] = $layer;
        }

        $hidden[] = new Dense(1, $this->l2Penalty, true, new Xavier2());

        $this->network = new FeedForward(
            new Placeholder1D($numInputs),
            $hidden,
            new Continuous($this->costFn),
            $this->optimizer
        );

        $this->network->initialize();

        $this->partial($dataset);
    }

    /**
     * Continue training from the current network state.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @throws RuntimeException
     */
    public function partial(Dataset $dataset) : void
    {
        if (!$this->network) {
            $this->train($dataset);

            return;
        }

        SpecificationChain::with([
            new DatasetIsLabeled($dataset),
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
            new LabelsAreCompatibleWithLearner($dataset, $this),
            new DatasetHasDimensionality(
                $dataset,
                $this->network->input()->width()
            ),
        ])->check();

        if ($this->logger) {
            $this->logger->info("Training $this");
            $numParams = number_format($this->network->numParams());
            $this->logger->info("{$numParams} trainable parameters");
        }

        [$testing, $training] = $dataset->randomize()->split($this->holdOut);

        [$minScore, $maxScore] = $this->metric->range()->list();

        $bestScore = $minScore;
        $bestEpoch = $numWorseEpochs = 0;
        $loss      = 0.0;
        $score     = $snapshot = null;
        $prevLoss  = INF;

        $this->scores = $this->losses = [];

        for ($epoch = 1; $epoch <= $this->epochs; ++$epoch) {
            $batches = $training->randomize()->batch($this->batchSize);
            $loss    = 0.0;

            foreach ($batches as $batch) {
                $loss += $this->network->roundtrip($batch);
            }

            $loss /= count($batches);

            $this->losses[$epoch] = $loss;

            if (is_nan($loss)) {
                if ($this->logger) {
                    $this->logger->warning('Numerical instability detected');
                }

                break;
            }

            $lossChange = abs($prevLoss - $loss);

            if (!$testing->empty()) {
                $predictions = $this->predict($testing);
                $score       = $this->metric->score($predictions, $testing->labels());

                $this->scores[$epoch] = $score;
            }

            if ($this->logger) {
                $dir = $loss < $prevLoss ? '↓' : '↑';
                $this->logger->info(
                    "Epoch: $epoch, "
                    . "{$this->costFn}: $loss, "
                    . "Change: {$dir}{$lossChange}, "
                    . "{$this->metric}: " . ($score ?? 'N/A')
                );
            }

            if (isset($score)) {
                if ($score >= $maxScore) {
                    break;
                }

                if ($score > $bestScore) {
                    $bestScore      = $score;
                    $bestEpoch      = $epoch;
                    $snapshot       = Snapshot::take($this->network);
                    $numWorseEpochs = 0;
                } else {
                    ++$numWorseEpochs;
                }

                if ($numWorseEpochs >= $this->window) {
                    break;
                }
            }

            if ($lossChange < $this->minChange) {
                break;
            }

            $prevLoss = $loss;
        }

        if ($snapshot && (end($this->scores) < $bestScore || is_nan($loss))) {
            $snapshot->restore();

            if ($this->logger) {
                $this->logger->info("Model state restored to epoch $bestEpoch");
            }
        }

        if ($this->logger) {
            $this->logger->info('Training complete');
        }
    }

    /**
     * Make predictions for each sample in the dataset.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<float>
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->network) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->network->input()->width())->check();

        $activations = $this->network->infer($dataset);

        return array_column($activations->asArray(), 0);
    }

    /**
     * Export the network graph in Graphviz dot format.
     *
     * @throws RuntimeException
     * @return Encoding
     */
    public function exportGraphviz() : Encoding
    {
        if (!$this->network) {
            throw new RuntimeException('Must train network first.');
        }

        return $this->network->exportGraphviz();
    }

    /**
     * @return mixed[]
     */
    public function __serialize() : array
    {
        $properties = get_object_vars($this);

        unset($properties['losses'], $properties['scores'], $properties['logger']);

        return $properties;
    }

    /**
     * @return string
     */
    public function __toString() : string
    {
        return 'RNN Regressor (' . Params::stringify($this->params()) . ')';
    }
}
