<?php

declare(strict_types=1);

namespace Rubix\ML\Classifiers;

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
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Traits\LoggerAware;
use Rubix\ML\NeuralNet\Snapshot;
use Rubix\ML\NeuralNet\FeedForward;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Binary;
use Rubix\ML\NeuralNet\Layers\Dropout;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\NeuralNet\Layers\LSTM;
use Rubix\ML\NeuralNet\Layers\BlasLSTM;
use Rubix\ML\NeuralNet\Layers\Placeholder1D;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Initializers\Xavier2;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
use Rubix\ML\NeuralNet\CostFunctions\ClassificationLoss;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\DatasetIsLabeled;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Blas\BLASFFI;
use Generator;

use function is_nan;
use function count;
use function array_unique;
use function array_values;
use function number_format;

/**
 * LSTM Classifier
 *
 * A binary classification network built on an LSTM hidden layer followed by a
 * Dropout regulariser and a single sigmoid output neuron trained with Binary
 * Cross-Entropy loss.  When libBLAS is available (PHP-FFI, libopenblas, etc.)
 * the LSTM layer is automatically replaced with BlasLSTM for substantially
 * faster training.
 *
 * Architecture
 * ────────────
 *   Placeholder1D(seqLen × inputSize)
 *     → BlasLSTM / LSTM (hiddenSize)
 *     → Dropout (ratio)
 *     → Dense (1)
 *     → Binary (CrossEntropy)
 *
 * Dataset format
 * ──────────────
 * Each sample must be a flat float vector of length (sequenceLength × inputSize):
 *   [step0_feat0, …, step0_featI−1, step1_feat0, …, step_{T−1}_feat_{I−1}]
 * Labels must be one of exactly two string class names.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 */
class LSTMClassifier implements Estimator, Learner, Online, Verbose, Persistable
{
    use AutotrackRevisions, LoggerAware;

    /** @var positive-int */
    protected int $sequenceLength;

    /** @var positive-int */
    protected int $inputSize;

    /** @var positive-int */
    protected int $hiddenSize;

    /** @var float */
    protected float $dropoutRatio;

    /** @var Hidden[] */
    protected array $extraHidden;

    /** @var positive-int */
    protected int $batchSize;

    /** @var Optimizer */
    protected Optimizer $optimizer;

    /** @var float */
    protected float $l2Penalty;

    /** @var int<0,max> */
    protected int $epochs;

    /** @var float */
    protected float $minChange;

    /** @var positive-int */
    protected int $window;

    /** @var float */
    protected float $holdOut;

    /** @var ClassificationLoss */
    protected ClassificationLoss $costFn;

    /** @var Metric */
    protected Metric $metric;

    /** @var FeedForward|null */
    protected ?FeedForward $network = null;

    /**
     * The two class labels discovered during training: [class0, class1].
     * Sigmoid output ≥ 0.5  → classes[1], else → classes[0].
     *
     * @var string[]
     */
    protected array $classes = [];

    /** @var float[]|null */
    protected ?array $scores = null;

    /** @var float[]|null */
    protected ?array $losses = null;

    /**
     * @param int                    $sequenceLength  Time steps per sample
     * @param int                    $inputSize       Features per time step
     * @param int                    $hiddenSize      LSTM hidden units
     * @param float                  $dropoutRatio    Dropout ratio after LSTM (0 = disabled)
     * @param Hidden[]               $extraHidden     Extra Dense layers between LSTM and output
     * @param int                    $batchSize
     * @param Optimizer|null         $optimizer
     * @param float                  $l2Penalty       L2 regularisation for the Dense layer
     * @param int                    $epochs
     * @param float                  $minChange
     * @param int                    $window          Early-stopping patience (epochs)
     * @param float                  $holdOut         Validation fraction [0, 0.5]
     * @param ClassificationLoss|null $costFn         Defaults to CrossEntropy (BCE)
     * @param Metric|null            $metric          Defaults to Accuracy
     * @throws InvalidArgumentException
     */
    public function __construct(
        int $sequenceLength = 20,
        int $inputSize = 6,
        int $hiddenSize = 64,
        float $dropoutRatio = 0.2,
        array $extraHidden = [],
        int $batchSize = 32,
        ?Optimizer $optimizer = null,
        float $l2Penalty = 1e-4,
        int $epochs = 100,
        float $minChange = 1e-4,
        int $window = 5,
        float $holdOut = 0.1,
        ?ClassificationLoss $costFn = null,
        ?Metric $metric = null
    ) {
        if ($sequenceLength < 1) {
            throw new InvalidArgumentException(
                "Sequence length must be at least 1, $sequenceLength given."
            );
        }

        if ($inputSize < 1) {
            throw new InvalidArgumentException(
                "Input size must be at least 1, $inputSize given."
            );
        }

        if ($hiddenSize < 1) {
            throw new InvalidArgumentException(
                "Hidden size must be at least 1, $hiddenSize given."
            );
        }

        if ($dropoutRatio < 0.0 || $dropoutRatio >= 1.0) {
            throw new InvalidArgumentException(
                "Dropout ratio must be in [0, 1), $dropoutRatio given."
            );
        }

        if ($batchSize < 1) {
            throw new InvalidArgumentException(
                "Batch size must be at least 1, $batchSize given."
            );
        }

        if ($l2Penalty < 0.0) {
            throw new InvalidArgumentException(
                "L2 penalty must be non-negative, $l2Penalty given."
            );
        }

        if ($epochs < 0) {
            throw new InvalidArgumentException(
                "Epochs must be non-negative, $epochs given."
            );
        }

        if ($minChange < 0.0) {
            throw new InvalidArgumentException(
                "Min change must be non-negative, $minChange given."
            );
        }

        if ($window < 1) {
            throw new InvalidArgumentException(
                "Window must be at least 1, $window given."
            );
        }

        if ($holdOut < 0.0 || $holdOut > 0.5) {
            throw new InvalidArgumentException(
                "Hold-out fraction must be between 0 and 0.5, $holdOut given."
            );
        }

        $this->sequenceLength = $sequenceLength;
        $this->inputSize      = $inputSize;
        $this->hiddenSize     = $hiddenSize;
        $this->dropoutRatio   = $dropoutRatio;
        $this->extraHidden    = $extraHidden;
        $this->batchSize      = $batchSize;
        $this->optimizer      = $optimizer ?? new Adam();
        $this->l2Penalty      = $l2Penalty;
        $this->epochs         = $epochs;
        $this->minChange      = $minChange;
        $this->window         = $window;
        $this->holdOut        = $holdOut;
        $this->costFn         = $costFn ?? new CrossEntropy();
        $this->metric         = $metric ?? new Accuracy();
    }

    /** @return EstimatorType */
    public function type(): EstimatorType
    {
        return EstimatorType::classifier();
    }

    /** @return list<DataType> */
    public function compatibility(): array
    {
        return [DataType::continuous()];
    }

    /** @return mixed[] */
    public function params(): array
    {
        return [
            'sequence length' => $this->sequenceLength,
            'input size'      => $this->inputSize,
            'hidden size'     => $this->hiddenSize,
            'dropout ratio'   => $this->dropoutRatio,
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

    /** @return bool */
    public function trained(): bool
    {
        return isset($this->network);
    }

    /** @return string[] The two class labels (class0 → sigmoid ≈ 0, class1 → sigmoid ≈ 1) */
    public function classes(): array
    {
        return $this->classes;
    }

    /** @return Generator<mixed[]> */
    public function steps(): Generator
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

    /** @return float[]|null */
    public function scores(): ?array
    {
        return $this->scores;
    }

    /** @return float[]|null */
    public function losses(): ?array
    {
        return $this->losses;
    }

    /** @return FeedForward|null */
    public function network(): ?FeedForward
    {
        return $this->network;
    }

    /**
     * Train the classifier on a labeled dataset.
     *
     * @param Labeled $dataset
     * @throws InvalidArgumentException
     */
    public function train(Dataset $dataset): void
    {
        DatasetIsLabeled::with($dataset)->check();
        DatasetIsNotEmpty::with($dataset)->check();

        /** @var Labeled $dataset */
        $labels = $dataset->labels();

        // Discover and store class labels
        $classes = array_values(array_unique($labels));

        if (count($classes) !== 2) {
            throw new InvalidArgumentException(
                'LSTMClassifier requires exactly 2 classes, '
                . count($classes) . ' found.'
            );
        }

        sort($classes);  // deterministic ordering
        $this->classes = $classes;

        // Build network
        $numInputs = $this->sequenceLength * $this->inputSize;

        // Use BLAS-accelerated LSTM when the library is available
        $lstmLayer = BLASFFI::isAvailable()
            ? new BlasLSTM($this->hiddenSize, $this->inputSize)
            : new LSTM($this->hiddenSize, $this->inputSize);

        $hidden = [$lstmLayer];

        if ($this->dropoutRatio > 0.0) {
            $hidden[] = new Dropout($this->dropoutRatio);
        }

        foreach ($this->extraHidden as $layer) {
            $hidden[] = $layer;
        }

        $hidden[] = new Dense(1, $this->l2Penalty, true, new Xavier2());

        $this->network = new FeedForward(
            new Placeholder1D($numInputs),
            $hidden,
            new Binary($this->classes, $this->costFn),
            $this->optimizer
        );

        $this->network->initialize();

        $this->partial($dataset);
    }

    /**
     * Continue training on new data.
     *
     * @param Labeled $dataset
     * @throws RuntimeException
     */
    public function partial(Dataset $dataset): void
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
            $blasTag = BLASFFI::isAvailable() ? 'BLAS accelerated' : 'PHP fallback';
            $this->logger->info("LSTM path: $blasTag");
        }

        [$testing, $training] = $dataset->randomize()->split($this->holdOut);

        [$minScore, $maxScore] = $this->metric->range()->list();

        $bestScore      = $minScore;
        $bestEpoch      = $numWorseEpochs = 0;
        $loss           = 0.0;
        $score          = $snapshot = null;
        $prevLoss       = INF;

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
                    . "{$this->costFn}: " . round($loss, 6) . ", "
                    . "Change: {$dir}" . round($lossChange, 6) . ", "
                    . "{$this->metric}: " . ($score !== null ? round($score, 4) : 'N/A')
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
     * Predict class labels for each sample.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<string>
     */
    public function predict(Dataset $dataset): array
    {
        if (!$this->network) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->network->input()->width())->check();

        // infer() returns (B, 1) matrix after internal transpose
        $activations = $this->network->infer($dataset);
        $probas      = array_column($activations->asArray(), 0);

        $class0 = $this->classes[0];
        $class1 = $this->classes[1];

        return array_map(
            static fn (float $p): string => $p >= 0.5 ? $class1 : $class0,
            $probas
        );
    }

    /**
     * Return sigmoid probabilities (probability of classes[1]).
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<float>
     */
    public function proba(Dataset $dataset): array
    {
        if (!$this->network) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->network->input()->width())->check();

        $activations = $this->network->infer($dataset);

        return array_column($activations->asArray(), 0);
    }

    /** @return mixed[] */
    public function __serialize(): array
    {
        $properties = get_object_vars($this);

        unset($properties['losses'], $properties['scores'], $properties['logger']);

        return $properties;
    }

    /** @return string */
    public function __toString(): string
    {
        return 'LSTM Classifier (' . Params::stringify($this->params()) . ')';
    }
}
