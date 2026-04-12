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
use Rubix\ML\NeuralNet\Layers\Conv1D;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Layers\Continuous;
use Rubix\ML\NeuralNet\Layers\Placeholder1D;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Initializers\Xavier2;
use Rubix\ML\NeuralNet\ActivationFunctions\ReLU;
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
 * CNN Regressor
 *
 * A 1-D Convolutional Neural Network for time-series regression. The architecture
 * stacks one or more `Conv1D` layers followed by optional dense layers and a single
 * continuous output neuron, using the same mini-batch gradient-descent training
 * loop as `MLPRegressor`.
 *
 * **Dataset format**: each sample must be a flat feature vector of length
 * (sequenceLength × inputChannels) where values are arranged as:
 *   [step0_ch0, step0_ch1, …, step1_ch0, …, stepT_ch0, …]
 *
 * This matches the natural output of a sliding-window transformer applied to a
 * multi-channel time series.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      The Rubix ML Community
 */
class CNNRegressor implements Estimator, Learner, Online, Verbose, Persistable
{
    use AutotrackRevisions, LoggerAware;

    /**
     * Length of the input sequence (time steps).
     *
     * @var positive-int
     */
    protected int $sequenceLength;

    /**
     * Number of input channels per time step (features per step).
     *
     * @var positive-int
     */
    protected int $inputChannels;

    /**
     * Number of Conv1D filters in the first (and only default) convolutional block.
     *
     * @var positive-int
     */
    protected int $numFilters;

    /**
     * Kernel size of the Conv1D layer.
     *
     * @var positive-int
     */
    protected int $kernelSize;

    /**
     * Additional hidden Dense layers placed after the Conv1D block.
     *
     * @var Hidden[]
     */
    protected array $denseHidden;

    /**
     * Mini-batch size.
     *
     * @var positive-int
     */
    protected int $batchSize;

    /**
     * Gradient-descent optimizer.
     *
     * @var Optimizer
     */
    protected Optimizer $optimizer;

    /**
     * L2 regularization applied to the final Dense layer.
     *
     * @var float
     */
    protected float $l2Penalty;

    /**
     * Maximum training epochs.
     *
     * @var int<0,max>
     */
    protected int $epochs;

    /**
     * Minimum loss change to continue training (early-stop criterion).
     *
     * @var float
     */
    protected float $minChange;

    /**
     * Number of epochs without validation improvement before stopping.
     *
     * @var positive-int
     */
    protected int $window;

    /**
     * Fraction of training data reserved for validation.
     *
     * @var float
     */
    protected float $holdOut;

    /**
     * Regression loss function.
     *
     * @var RegressionLoss
     */
    protected RegressionLoss $costFn;

    /**
     * Validation metric.
     *
     * @var Metric
     */
    protected Metric $metric;

    /**
     * The underlying FeedForward network.
     *
     * @var FeedForward|null
     */
    protected ?FeedForward $network = null;

    /**
     * Validation scores per epoch.
     *
     * @var float[]|null
     */
    protected ?array $scores = null;

    /**
     * Training losses per epoch.
     *
     * @var float[]|null
     */
    protected ?array $losses = null;

    /**
     * @param int              $sequenceLength  Time steps per sample
     * @param int              $inputChannels   Features per time step
     * @param int              $numFilters      Conv1D output filters
     * @param int              $kernelSize      Conv1D kernel width
     * @param Hidden[]         $denseHidden     Extra Dense layers after Conv1D
     * @param int              $batchSize       Mini-batch size
     * @param Optimizer|null   $optimizer
     * @param float            $l2Penalty       L2 regularisation
     * @param int              $epochs          Max training epochs
     * @param float            $minChange       Early-stop loss threshold
     * @param int              $window          Early-stop patience (epochs)
     * @param float            $holdOut         Validation hold-out ratio
     * @param RegressionLoss|null $costFn
     * @param Metric|null      $metric
     * @throws InvalidArgumentException
     */
    public function __construct(
        int $sequenceLength = 24,
        int $inputChannels = 1,
        int $numFilters = 32,
        int $kernelSize = 3,
        array $denseHidden = [],
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

        if ($inputChannels < 1) {
            throw new InvalidArgumentException("Input channels must be"
                . " at least 1, $inputChannels given.");
        }

        if ($numFilters < 1) {
            throw new InvalidArgumentException("Number of filters must be"
                . " at least 1, $numFilters given.");
        }

        if ($kernelSize < 1) {
            throw new InvalidArgumentException("Kernel size must be"
                . " at least 1, $kernelSize given.");
        }

        if ($batchSize < 1) {
            throw new InvalidArgumentException("Batch size must be"
                . " greater than 0, $batchSize given.");
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
        $this->inputChannels  = $inputChannels;
        $this->numFilters     = $numFilters;
        $this->kernelSize     = $kernelSize;
        $this->denseHidden    = $denseHidden;
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
            'input channels'  => $this->inputChannels,
            'num filters'     => $this->numFilters,
            'kernel size'     => $this->kernelSize,
            'dense hidden'    => $this->denseHidden,
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
     * Return an iterable progress table.
     *
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
     * Train the CNN on a labeled dataset.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     */
    public function train(Dataset $dataset) : void
    {
        DatasetIsNotEmpty::with($dataset)->check();

        $numInputs = $this->sequenceLength * $this->inputChannels;

        // Build the network layers
        $hidden = [
            new Conv1D(
                $this->numFilters,
                $this->kernelSize,
                $this->inputChannels
            ),
            new Activation(new ReLU()),
        ];

        foreach ($this->denseHidden as $layer) {
            $hidden[] = $layer;
        }

        // Final projection to 1 output
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
                    $bestScore       = $score;
                    $bestEpoch       = $epoch;
                    $snapshot        = Snapshot::take($this->network);
                    $numWorseEpochs  = 0;
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
     * Make continuous predictions for each sample in the dataset.
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
        return 'CNN Regressor (' . Params::stringify($this->params()) . ')';
    }
}
