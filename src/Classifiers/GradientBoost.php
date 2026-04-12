<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\RanksFeatures;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Stats;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Traits\LoggerAware;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Regressors\RegressionTree;
use Rubix\ML\Regressors\ExtraTreeRegressor;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\CrossValidation\Metrics\Metric;
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

use function count;
use function is_nan;
use function get_class;
use function in_array;
use function array_map;
use function array_fill;
use function array_flip;
use function array_slice;
use function array_values;
use function array_unique;
use function exp;
use function log;
use function max;
use function abs;
use function round;
use function get_object_vars;

/**
 * Gradient Boost Classifier
 *
 * A stage-wise additive ensemble that minimises a deviance loss via functional
 * gradient descent, fitting regression trees to the **negative gradients**
 * (pseudo-residuals) of the loss function at each boosting round.
 *
 * -------------------------------------------------------------------------
 * Loss functions
 * -------------------------------------------------------------------------
 * K = 2 — **Binomial Deviance** (binary cross-entropy on log-odds):
 *
 *   L = −(1/m) Σᵢ [ yᵢ log pᵢ + (1−yᵢ) log(1−pᵢ) ]
 *
 *   where pᵢ = σ(F(xᵢ)) and F accumulates the log-odds of the positive class.
 *
 * K > 2 — **Multinomial Deviance** (softmax cross-entropy):
 *
 *   L = −(1/m) Σᵢ log p_{yᵢ}(xᵢ)
 *
 *   where p_k = softmax_k(F₁, …, Fₖ) and each class k maintains its own
 *   score accumulator Fₖ.
 *
 * -------------------------------------------------------------------------
 * Newton–Raphson terminal-node update (key distinction from LogitBoost)
 * -------------------------------------------------------------------------
 * A plain regression tree fitted to the pseudo-residuals predicts the *mean*
 * residual in each leaf.  The theoretically optimal leaf value under Newton's
 * method is:
 *
 *   γⱼ = Σ_{i∈Rⱼ} rᵢ / Σ_{i∈Rⱼ} Hᵢ
 *
 * where Hᵢ = pᵢ(1−pᵢ) is the diagonal of the Hessian (second-order
 * curvature of the loss w.r.t. Fᵢ).  Because Rubix's RegressionTree does
 * not expose its leaf values for post-hoc modification, we approximate the
 * per-leaf Newton step with a **global Hessian scale**:
 *
 *   H = (1/m) Σᵢ pᵢ(1−pᵢ)          (scalar mean curvature)
 *
 * The tree's raw predictions (mean residual per leaf) are then divided by H
 * before being added to F, giving an effective update rate of `η / H` instead
 * of the plain learning rate η.  This scale is stored alongside each booster
 * so prediction is exact.
 *
 *   Binary:     F(x)  += (η / H)  · tree.predict(x)
 *   Multi-class: Fₖ(x) += (η / Hₖ) · treeₖ.predict(x)   for each class k
 *
 * This approximation converges faster than plain gradient descent (LogitBoost)
 * and produces better-calibrated probabilities.
 *
 * -------------------------------------------------------------------------
 * Trees per epoch
 * -------------------------------------------------------------------------
 *   K = 2:  **1 tree** per epoch fitted to the scalar residuals of the single
 *           log-odds accumulator F.  Predicting class[1] iff F(x) ≥ 0.
 *
 *   K > 2:  **K trees** per epoch, one per class, fitted to the per-class
 *           pseudo-residuals rᵢₖ = I(yᵢ = cₖ) − pₖ(xᵢ).
 *
 * -------------------------------------------------------------------------
 * JIT optimisation
 * -------------------------------------------------------------------------
 * - All mathematical arrays pre-allocated with `array_fill(0, $n, 0.0)`.
 * - Arrays that accumulate floats are strictly homogeneous (no mixed types).
 * - Large temporaries (`$pseudoDataset`, `$treePreds`, etc.) are `unset()`
 *   immediately after their last use to release heap memory before the next
 *   allocation, preventing out-of-memory errors on long training runs.
 *
 * References:
 * [1] J. H. Friedman. (2001). Greedy Function Approximation: A Gradient
 *     Boosting Machine.
 * [2] J. H. Friedman. (1999). Stochastic Gradient Boosting.
 * [3] J. H. Friedman et al. (2000). Additive Logistic Regression: A
 *     Statistical View of Boosting.
 * [4] Y. Wei et al. (2017). Early stopping for kernel boosting algorithms.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class GradientBoost implements Estimator, Learner, Probabilistic, RanksFeatures, Verbose, Persistable
{
    use AutotrackRevisions, LoggerAware;

    /**
     * Compatible weak-learner class names (must be regression-capable).
     *
     * @var class-string[]
     */
    public const COMPATIBLE_BOOSTERS = [
        RegressionTree::class,
        ExtraTreeRegressor::class,
    ];

    /**
     * Minimum subsample size to prevent degenerate single-sample trees.
     *
     * @var int
     */
    protected const MIN_SUBSAMPLE = 2;

    /**
     * Floor applied to the global Hessian H before division to prevent
     * the effective learning rate from exploding when the model is already
     * very confident (all pᵢ near 0 or 1).
     *
     * @var float
     */
    protected const MIN_HESSIAN = 1e-6;

    /**
     * Epsilon added inside log() calls to prevent log(0) = −∞ propagating
     * as NaN into the loss and gradient.
     *
     * @var float
     */
    protected const LOG_EPSILON = 1e-10;

    /**
     * The regression-tree prototype cloned once per class per epoch.
     *
     * @var Learner
     */
    protected Learner $booster;

    /**
     * The base learning rate η (shrinkage). The *effective* per-booster rate
     * is η / H (Newton step), stored in $scales; this property is retained
     * only for serialisation and display purposes.
     *
     * @var float
     */
    protected float $rate;

    /**
     * Fraction of training samples drawn (with replacement) for each booster.
     * Values < 1.0 introduce stochasticity and act as implicit regularisation.
     *
     * @var float
     */
    protected float $ratio;

    /**
     * Maximum number of boosting rounds (epochs). Each round adds 1 tree for
     * binary and K trees for multi-class.
     *
     * @var int<0,max>
     */
    protected int $epochs;

    /**
     * Minimum absolute decrease in training loss required to continue. Acts as
     * a convergence criterion that halts training when gains are negligible.
     *
     * @var float
     */
    protected float $minChange;

    /**
     * Number of consecutive epochs without improvement in the held-out metric
     * before early stopping is triggered (patience window).
     *
     * @var positive-int
     */
    protected int $window;

    /**
     * Fraction of training data reserved for validation and early stopping.
     * The split is stratified to preserve class balance.
     *
     * @var float
     */
    protected float $holdOut;

    /**
     * Classification metric used to score the held-out set each epoch.
     *
     * @var Metric
     */
    protected Metric $metric;

    /**
     * Fitted boosters indexed by [classIndex][epochIndex].
     *
     * For K = 2 (binary): shape is [1][T] — one class slot, T epoch slots.
     * For K > 2 (multi-class): shape is [K][T] — one slot per class.
     *
     * @var list<list<Learner>>|null
     */
    protected ?array $boosters = null;

    /**
     * Effective per-booster learning rates, indexed by [classIndex][epochIndex].
     *
     * Defined as `η / H_{k,t}` where H_{k,t} is the global Hessian computed
     * at epoch t for class k.  Stored alongside boosters so predict() applies
     * exactly the same Newton-scaled update without recomputing H.
     *
     * @var list<list<float>>|null
     */
    protected ?array $scales = null;

    /**
     * Initial log-odds value F₀ used to seed the binary log-odds accumulator.
     *
     * Computed as log(p / (1−p)) where p is the training-set proportion of the
     * positive class (classes[1]). Initialising at this value is mathematically
     * equivalent to the first "intercept" boosting step and accelerates
     * convergence compared to starting at 0.
     *
     * Set to null for multi-class problems (each Fₖ is initialised from $f0).
     *
     * @var float|null
     */
    protected ?float $z0 = null;

    /**
     * Per-class log-prior base scores for multi-class boosting.
     *
     * Fₖ₀ = log(countₖ / m) — the closed-form optimal constant model for
     * multinomial deviance. Initialising at the log-prior rather than 0 gives
     * minority classes large gradient signal from round 1 and prevents the
     * dominant class from monopolising early boosting rounds.
     *
     * Stored after training so predict() and proba() can reconstruct the same
     * starting accumulator without recomputing class proportions.
     *
     * Null for binary problems (z0 is used instead).
     *
     * @var float[]|null
     */
    protected ?array $f0 = null;

    /**
     * L2 regularisation term added to the per-class Hessian sum before
     * computing the Newton step scale (η / (Hₖ + λ/m)).
     *
     * Prevents the effective learning rate from exploding when a class
     * probability collapses toward 0 (Hₖ → 0), which is the primary cause
     * of minority-class oscillation in imbalanced datasets.  A value of 1–5
     * is a sensible default for most problems.
     *
     * @var float
     */
    protected float $lambda = 0.0;

    /**
     * Unique class labels in canonical order as discovered during training.
     *
     * @var list<string>|null
     */
    protected ?array $classes = null;

    /**
     * Number of feature columns in the training set. Used to validate incoming
     * predict() datasets.
     *
     * @var int<0,max>|null
     */
    protected ?int $featureCount = null;

    /**
     * Validation scores at each epoch from the last training session.
     *
     * @var float[]|null
     */
    protected ?array $scores = null;

    /**
     * Training loss at each epoch from the last training session.
     *
     * @var float[]|null
     */
    protected ?array $losses = null;

    /**
     * @param Learner|null $booster     Weak learner prototype (default: shallow RegressionTree of height 3).
     * @param float        $rate        Base learning rate η (0 < rate).
     * @param float        $ratio       Subsample ratio (0 < ratio ≤ 1).
     * @param int          $epochs      Maximum boosting rounds (≥ 0).
     * @param float        $minChange   Minimum loss change to continue (≥ 0).
     * @param int          $window      Early-stopping patience in epochs (≥ 1).
     * @param float        $holdOut     Validation set fraction (0 ≤ holdOut ≤ 0.5).
     * @param Metric|null  $metric      Validation metric (default: Accuracy).
     * @param float        $lambda      L2 regularisation on the Hessian (≥ 0). Prevents Newton-step
     *                                  explosion for minority classes in imbalanced multiclass problems.
     * @throws InvalidArgumentException
     */
    public function __construct(
        ?Learner $booster = null,
        float $rate = 0.1,
        float $ratio = 0.5,
        int $epochs = 1000,
        float $minChange = 1e-4,
        int $window = 5,
        float $holdOut = 0.1,
        ?Metric $metric = null,
        float $lambda = 0.0
    ) {
        if ($booster and !in_array(get_class($booster), self::COMPATIBLE_BOOSTERS)) {
            throw new InvalidArgumentException('Booster is not compatible'
                . ' with the ensemble.');
        }

        if ($rate <= 0.0) {
            throw new InvalidArgumentException('Learning rate must be'
                . " greater than 0, $rate given.");
        }

        if ($ratio <= 0.0 or $ratio > 1.0) {
            throw new InvalidArgumentException('Ratio must be'
                . " between 0 and 1, $ratio given.");
        }

        if ($epochs < 0) {
            throw new InvalidArgumentException('Number of epochs'
                . " must be greater than 0, $epochs given.");
        }

        if ($minChange < 0.0) {
            throw new InvalidArgumentException('Minimum change must be'
                . " greater than 0, $minChange given.");
        }

        if ($window < 1) {
            throw new InvalidArgumentException('Window must be'
                . " greater than 0, $window given.");
        }

        if ($holdOut < 0.0 or $holdOut > 0.5) {
            throw new InvalidArgumentException('Hold out ratio must be'
                . " between 0 and 0.5, $holdOut given.");
        }

        if ($lambda < 0.0) {
            throw new InvalidArgumentException('Lambda must be'
                . " greater than or equal to 0, $lambda given.");
        }

        if ($metric) {
            EstimatorIsCompatibleWithMetric::with($this, $metric)->check();
        }

        $this->booster   = $booster ?? new RegressionTree(3);
        $this->rate      = $rate;
        $this->ratio     = $ratio;
        $this->epochs    = $epochs;
        $this->minChange = $minChange;
        $this->window    = $window;
        $this->holdOut   = $holdOut;
        $this->metric    = $metric ?? new Accuracy();
        $this->lambda    = $lambda;
    }

    /**
     * Return the estimator type.
     *
     * @internal
     *
     * @return EstimatorType
     */
    public function type() : EstimatorType
    {
        return EstimatorType::classifier();
    }

    /**
     * Return the data types that this estimator is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function compatibility() : array
    {
        return $this->booster->compatibility();
    }

    /**
     * Return the hyper-parameter settings in an associative array.
     *
     * @internal
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'booster'    => $this->booster,
            'rate'       => $this->rate,
            'ratio'      => $this->ratio,
            'epochs'     => $this->epochs,
            'min change' => $this->minChange,
            'window'     => $this->window,
            'hold out'   => $this->holdOut,
            'metric'     => $this->metric,
            'lambda'     => $this->lambda,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return isset($this->boosters, $this->scales, $this->classes, $this->featureCount);
    }

    /**
     * Return an iterable progress table with the steps from the last training session.
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
     * Return the validation scores at each epoch from the last training session.
     *
     * @return float[]|null
     */
    public function scores() : ?array
    {
        return $this->scores;
    }

    /**
     * Return the loss at each epoch from the last training session.
     *
     * @return float[]|null
     */
    public function losses() : ?array
    {
        return $this->losses;
    }

    /**
     * Train the estimator with a dataset.
     *
     * Dispatches to the binary (K=2) or multi-class (K>2) training path after
     * shared setup (validation, stratified split, target encoding).
     *
     * @param Labeled $dataset
     */
    public function train(Dataset $dataset) : void
    {
        SpecificationChain::with([
            new DatasetIsLabeled($dataset),
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
            new LabelsAreCompatibleWithLearner($dataset, $this),
        ])->check();

        /** @var list<string> $classes */
        $classes = array_values(array_unique($dataset->labels()));

        if ($this->logger) {
            $this->logger->info("Training $this");
        }

        // Stratified split preserves class proportions in both partitions.
        [$testing, $training] = $dataset->stratifiedSplit($this->holdOut);

        [$minScore, $maxScore] = $this->metric->range()->list();

        [$m, $n] = $training->shape();

        $numClasses = count($classes);

        // Map label string → integer index (0…K−1).
        $classMap = array_flip($classes);

        // Cache sample array once — Labeled::quick() wraps without copying.
        $trainingSamples = $training->samples();

        // Subsample size for stochastic boosting.
        $p = max(self::MIN_SUBSAMPLE, (int) round($this->ratio * $m));

        // Uniform weights to start; updated each epoch to concentrate future
        // draws on hard examples (by absolute residual magnitude).
        /** @var float[] $weights */
        $weights = array_fill(0, $m, 1.0 / $m);

        // Initialise shared training-history state.
        $this->classes      = $classes;
        $this->featureCount = $n;
        $this->scores       = [];
        $this->losses       = [];

        $bestScore      = $minScore;
        $bestEpoch      = 0;
        $numWorseEpochs = 0;
        $score          = null;
        $prevLoss       = INF;

        // =====================================================================
        // Dispatch: separate code paths for binary vs multi-class to avoid
        // unnecessary overhead (binary needs only 1 tree/epoch and a 1-D z
        // accumulator, while multi-class needs K trees/epoch and a 2-D F matrix).
        // =====================================================================

        if ($numClasses === 2) {
            $this->trainBinary(
                $training, $testing, $trainingSamples,
                $classes, $classMap,
                $m, $n, $p,
                $weights,
                $minScore, $maxScore,
                $bestScore, $bestEpoch, $numWorseEpochs,
                $score, $prevLoss
            );
        } else {
            $this->trainMulticlass(
                $training, $testing, $trainingSamples,
                $classes, $classMap,
                $m, $n, $p, $numClasses,
                $weights,
                $minScore, $maxScore,
                $bestScore, $bestEpoch, $numWorseEpochs,
                $score, $prevLoss
            );
        }

        unset($trainingSamples, $weights);

        if ($this->logger) {
            $this->logger->info('Training complete.');
        }
    }

    // =========================================================================
    // Binary training path  (K = 2)
    // =========================================================================

    /**
     * Core boosting loop for K = 2 (Binomial Deviance).
     *
     * One tree is trained per epoch on the scalar pseudo-residuals of the
     * single log-odds accumulator z:
     *
     *   rᵢ   = yᵢ − σ(z(xᵢ))          [negative gradient of BinomDev]
     *   H    = (1/m) Σᵢ σ(z)(1 − σ(z)) [global diagonal Hessian]
     *   z(x) += (η / H) · tree.predict(x)
     *
     * The effective rate η/H is stored per-epoch so predict() can reconstruct
     * z without recomputing H.
     *
     * @param Labeled   $training
     * @param Labeled   $testing
     * @param array     $trainingSamples
     * @param array     $classes
     * @param array     $classMap
     * @param int       $m
     * @param int       $n
     * @param int       $p
     * @param float[]   $weights
     * @param float     $minScore
     * @param float     $maxScore
     * @param float     $bestScore
     * @param int       $bestEpoch
     * @param int       $numWorseEpochs
     * @param float|null $score
     * @param float     $prevLoss
     */
    protected function trainBinary(
        Labeled $training,
        Labeled $testing,
        array $trainingSamples,
        array $classes,
        array $classMap,
        int $m,
        int $n,
        int $p,
        array &$weights,
        float $minScore,
        float $maxScore,
        float $bestScore,
        int $bestEpoch,
        int $numWorseEpochs,
        ?float $score,
        float $prevLoss
    ) : void {
        // -----------------------------------------------------------------
        // Encode labels: classes[0] → 0.0 (negative), classes[1] → 1.0 (positive).
        // Pre-allocated as a homogeneous float array.
        // -----------------------------------------------------------------
        /** @var float[] $targets */
        $targets = array_fill(0, $m, 0.0);

        foreach ($training->labels() as $i => $label) {
            $targets[$i] = (float) $classMap[$label];
        }

        // -----------------------------------------------------------------
        // Initialise the log-odds accumulator at the unconditional log-odds
        // of the positive class.  This is the closed-form optimal constant
        // model (the "intercept" step of gradient boosting) and is faster to
        // converge from than zero.
        //
        //   z₀ = log(p̄ / (1 − p̄))   where p̄ = fraction of class[1] samples
        // -----------------------------------------------------------------
        $pBar = array_sum($targets) / $m;
        $pBar = max(self::LOG_EPSILON, min(1.0 - self::LOG_EPSILON, $pBar));

        $z0 = log($pBar / (1.0 - $pBar));

        /** @var float[] $z   Log-odds accumulator, one value per training sample. */
        $z = array_fill(0, $m, $z0);

        /** @var float[] $out  Sigmoid probabilities of the positive class. */
        $out = array_map('Rubix\ML\sigmoid', $z);

        // Scalar pseudo-residuals (overwritten in-place each epoch).
        /** @var float[] $residuals */
        $residuals = array_fill(0, $m, 0.0);

        // Log-odds accumulator for the held-out set (if any).
        $mTest = $testing->numSamples();

        /** @var float[]|null $zTest */
        $zTest = null;

        if ($mTest > 0) {
            $zTest = array_fill(0, $mTest, $z0);
        } elseif ($this->logger) {
            $this->logger->notice('Insufficient validation data, '
                . 'some features are disabled.');
        }

        // One booster/scale slot for the single binary class.
        $this->boosters = [[]];
        $this->scales   = [[]];
        $this->z0       = $z0;

        [$classA, $classB] = $classes;   // classA = negative, classB = positive

        // =================================================================
        // Binary boosting loop
        // =================================================================
        for ($epoch = 1; $epoch <= $this->epochs; ++$epoch) {

            // -------------------------------------------------------------
            // Step 1: Compute pseudo-residuals and Hessian.
            //
            // rᵢ = yᵢ − pᵢ   (negative gradient of Binomial Deviance w.r.t. Fᵢ)
            // Hᵢ = pᵢ(1−pᵢ)  (diagonal of the Hessian)
            //
            // Both are computed in a single pass over m samples.
            // The residuals array is overwritten in-place — no new allocation.
            // -------------------------------------------------------------
            // -------------------------------------------------------------
            // Steps 1 + 2 fused: single O(m) pass computes residuals,
            // Hessian accumulator, and binary cross-entropy loss together.
            // Each element of $out and $targets is read exactly once,
            // maximising CPU cache utilisation.
            // -------------------------------------------------------------
            $hessSum = 0.0;
            $loss    = 0.0;

            for ($i = 0; $i < $m; ++$i) {
                $p = $out[$i];
                $y = $targets[$i];
                $residuals[$i] = $y - $p;
                $hessSum      += $p * (1.0 - $p);
                $loss         -= $y * log(max($p, self::LOG_EPSILON))
                               + (1.0 - $y) * log(max(1.0 - $p, self::LOG_EPSILON));
            }

            $loss /= $m;

            // Global Hessian approximation: mean curvature over the training set.
            $H = max($hessSum / $m, self::MIN_HESSIAN);

            $lossChange = abs($prevLoss - $loss);

            $this->losses[$epoch] = $loss;

            // -------------------------------------------------------------
            // Step 3: Score the held-out set on the CURRENT z accumulator
            //         (before this epoch's tree is added).
            // -------------------------------------------------------------
            if (isset($zTest)) {
                $testPreds = [];

                foreach ($zTest as $zVal) {
                    $testPreds[] = $zVal >= 0.0 ? $classB : $classA;
                }

                $score = $this->metric->score($testPreds, $testing->labels());

                $this->scores[$epoch] = $score;

                unset($testPreds);
            }

            // -------------------------------------------------------------
            // Step 4: Logging.
            // -------------------------------------------------------------
            if ($this->logger) {
                $dir     = $loss < $prevLoss ? '↓' : '↑';
                $message = "Epoch: $epoch, "
                    . "Binomial Deviance: $loss, "
                    . "Loss Change: {$dir}{$lossChange}, "
                    . "Global Hessian: $H, "
                    . "{$this->metric}: " . ($score ?? 'N/A');

                $this->logger->info($message);
            }

            // -------------------------------------------------------------
            // Step 5: Early-stopping guards.
            // -------------------------------------------------------------
            if (is_nan($loss)) {
                if ($this->logger) {
                    $this->logger->warning('Numerical instability detected.');
                }

                break;
            }

            if (isset($score)) {
                if ($score >= $maxScore) {
                    break;
                }

                if ($score > $bestScore) {
                    $bestScore      = $score;
                    $bestEpoch      = $epoch;
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

            // -------------------------------------------------------------
            // Step 6: Fit one tree to the pseudo-residuals.
            //
            //   a. Wrap training samples + residuals in a Labeled dataset.
            //   b. Draw a weighted subsample (with replacement).
            //   c. Clone + train the booster prototype.
            //   d. Compute effective Newton rate: scale = η / H.
            //   e. Update z on training set: z[i] += scale · pred[i].
            //   f. Recompute sigmoid probabilities from updated z.
            //   g. Mirror the update on the held-out z accumulator.
            //   h. Unset temporaries immediately.
            // -------------------------------------------------------------
            $pseudoDataset = Labeled::quick($trainingSamples, $residuals);
            $subset        = $pseudoDataset->randomWeightedSubsetWithReplacement($p, $weights);

            $booster = clone $this->booster;
            $booster->train($subset);

            unset($pseudoDataset, $subset);

            // Newton-scaled effective rate for this epoch.
            $scale = $this->rate / $H;

            $this->boosters[0][] = $booster;
            $this->scales[0][]   = $scale;

            // Update training z accumulator.
            $treePreds = $booster->predict($training);

            for ($i = 0; $i < $m; ++$i) {
                $z[$i] += $scale * $treePreds[$i];
            }

            // Recompute probabilities in-place — no new array allocation per epoch.
            for ($i = 0; $i < $m; ++$i) {
                $out[$i] = \Rubix\ML\sigmoid($z[$i]);
            }

            unset($treePreds);

            // Mirror the Newton update on the held-out accumulator.
            if (isset($zTest)) {
                $testTreePreds = $booster->predict($testing);

                for ($i = 0; $i < $mTest; ++$i) {
                    $zTest[$i] += $scale * $testTreePreds[$i];
                }

                unset($testTreePreds);
            }

            // -------------------------------------------------------------
            // Step 7: Update sample weights for the next epoch's subsample draw.
            //
            // Weight = |rᵢ| — samples with larger residuals were harder to
            // classify and receive more weight in the next draw.
            // Overwrite in-place (no new array allocation).
            // -------------------------------------------------------------
            for ($i = 0; $i < $m; ++$i) {
                $weights[$i] = abs($residuals[$i]);
            }

            $prevLoss = $loss;
        }
        // =================================================================
        // End binary boosting loop
        // =================================================================

        // Best-epoch restoration: trim booster and scale lists back to the
        // epoch that achieved the best validation score.
        if ($this->scores and end($this->scores) < $bestScore) {
            $this->boosters[0] = array_slice($this->boosters[0], 0, $bestEpoch);
            $this->scales[0]   = array_slice($this->scales[0], 0, $bestEpoch);

            if ($this->logger) {
                $this->logger->info("Model state restored to epoch $bestEpoch.");
            }
        }

        unset($z, $out, $residuals, $targets, $zTest);
    }

    // =========================================================================
    // Multi-class training path  (K > 2)
    // =========================================================================

    /**
     * Core boosting loop for K > 2 (Multinomial Deviance).
     *
     * K trees are trained per epoch, one per class, on the per-class
     * pseudo-residuals of the softmax probability estimates:
     *
     *   rᵢₖ = I(yᵢ = cₖ) − pₖ(xᵢ)         [negative gradient]
     *   Hₖ  = (1/m) Σᵢ pₖ(xᵢ)(1 − pₖ(xᵢ)) [per-class global Hessian]
     *   Fₖ(x) += (η / Hₖ) · treeₖ.predict(x)
     *
     * @param Labeled    $training
     * @param Labeled    $testing
     * @param array      $trainingSamples
     * @param array      $classes
     * @param array      $classMap
     * @param int        $m
     * @param int        $n
     * @param int        $p
     * @param int        $numClasses
     * @param float[]    $weights
     * @param float      $minScore
     * @param float      $maxScore
     * @param float      $bestScore
     * @param int        $bestEpoch
     * @param int        $numWorseEpochs
     * @param float|null $score
     * @param float      $prevLoss
     */
    protected function trainMulticlass(
        Labeled $training,
        Labeled $testing,
        array $trainingSamples,
        array $classes,
        array $classMap,
        int $m,
        int $n,
        int $p,
        int $numClasses,
        array &$weights,
        float $minScore,
        float $maxScore,
        float $bestScore,
        int $bestEpoch,
        int $numWorseEpochs,
        ?float $score,
        float $prevLoss
    ) : void {
        // -----------------------------------------------------------------
        // Pre-compute integer class index for each training sample.
        // -----------------------------------------------------------------
        /** @var int[] $classIndices */
        $classIndices = array_fill(0, $m, 0);

        foreach ($training->labels() as $i => $label) {
            $classIndices[$i] = (int) $classMap[$label];
        }

        // Binary target matrix: $targets[$k][$i] = 1.0 iff sample i is class k.
        // Pre-allocated as K homogeneous float arrays.
        /** @var list<float[]> $targets */
        $targets = [];

        for ($k = 0; $k < $numClasses; ++$k) {
            $targets[$k] = array_fill(0, $m, 0.0);
        }

        for ($i = 0; $i < $m; ++$i) {
            $targets[$classIndices[$i]][$i] = 1.0;
        }

        // Per-class log-prior base scores: Fₖ₀ = log(countₖ / m).
        // This is the closed-form optimal constant model for multinomial deviance
        // and is the multiclass analogue of the binary log-odds initialisation.
        // Starting here instead of 0 gives minority classes strong gradient
        // signal (rᵢₖ = 1 − pₖ₀ ≈ 1 − countₖ/m) from round 1, preventing
        // the dominant class from monopolising early boosting rounds.
        /** @var int[] $classCounts */
        $classCounts = array_fill(0, $numClasses, 0);

        for ($i = 0; $i < $m; ++$i) {
            ++$classCounts[$classIndices[$i]];
        }

        /** @var float[] $logPriors */
        $logPriors = array_fill(0, $numClasses, 0.0);

        for ($k = 0; $k < $numClasses; ++$k) {
            $logPriors[$k] = log(max($classCounts[$k], 1) / $m);
        }

        $this->f0 = $logPriors;

        // Per-class score accumulators F[k][i], seeded at the log-prior.
        /** @var list<float[]> $F */
        $F = [];

        for ($k = 0; $k < $numClasses; ++$k) {
            $F[$k] = array_fill(0, $m, $logPriors[$k]);
        }

        // Softmax probability matrix, overwritten in-place each epoch.
        /** @var list<float[]> $probs */
        $probs = [];

        for ($k = 0; $k < $numClasses; ++$k) {
            $probs[$k] = array_fill(0, $m, 0.0);
        }

        // Per-class pseudo-residual matrix, overwritten in-place each epoch.
        /** @var list<float[]> $residuals */
        $residuals = [];

        for ($k = 0; $k < $numClasses; ++$k) {
            $residuals[$k] = array_fill(0, $m, 0.0);
        }

        // Per-class score accumulators for the held-out set.
        $mTest = $testing->numSamples();

        /** @var list<float[]>|null $Ftest */
        $Ftest = null;

        if ($mTest > 0) {
            $Ftest = [];

            for ($k = 0; $k < $numClasses; ++$k) {
                $Ftest[$k] = array_fill(0, $mTest, $logPriors[$k]);
            }
        } elseif ($this->logger) {
            $this->logger->notice('Insufficient validation data, '
                . 'some features are disabled.');
        }

        // Initialise K booster/scale slots.
        $this->boosters = [];
        $this->scales   = [];
        $this->z0       = null;   // not used in the multi-class path

        for ($k = 0; $k < $numClasses; ++$k) {
            $this->boosters[$k] = [];
            $this->scales[$k]   = [];
        }

        // =================================================================
        // Multi-class boosting loop
        // =================================================================
        for ($epoch = 1; $epoch <= $this->epochs; ++$epoch) {

            // -------------------------------------------------------------
            // Step 1: Compute numerically stable softmax probabilities.
            //
            // For each sample i, shift by the per-sample max F value to
            // prevent exp() overflow (log-sum-exp identity).
            // All writes go directly into the pre-allocated $probs arrays.
            // -------------------------------------------------------------
            // -------------------------------------------------------------
            // Steps 1 + 2 fused: numerically stable softmax + loss + residuals
            // in a single O(m) pass over samples (K inner steps each).
            //
            // Division by expSum replaced with reciprocal multiply (one
            // division per sample instead of K divisions per sample).
            // Loss is accumulated inline, eliminating a second O(m) scan.
            // -------------------------------------------------------------
            $loss = 0.0;

            for ($i = 0; $i < $m; ++$i) {
                $maxF = $F[0][$i];

                for ($k = 1; $k < $numClasses; ++$k) {
                    if ($F[$k][$i] > $maxF) {
                        $maxF = $F[$k][$i];
                    }
                }

                $expSum = 0.0;

                for ($k = 0; $k < $numClasses; ++$k) {
                    $e             = exp($F[$k][$i] - $maxF);
                    $probs[$k][$i] = $e;
                    $expSum       += $e;
                }

                $invExpSum = 1.0 / $expSum;
                $ci        = $classIndices[$i];

                for ($k = 0; $k < $numClasses; ++$k) {
                    $probs[$k][$i] *= $invExpSum;
                }

                // Loss for this sample (uses the freshly normalised probability).
                $loss -= log(max($probs[$ci][$i], self::LOG_EPSILON));
            }

            $loss /= $m;

            // Residuals: rᵢₖ = I(yᵢ=cₖ) − pₖ(xᵢ). K-outer/m-inner is
            // cache-friendly since each $residuals[$k] and $targets[$k] row
            // is a contiguous PHP array read sequentially.
            for ($k = 0; $k < $numClasses; ++$k) {
                $tk = $targets[$k];
                $pk = $probs[$k];
                $rk = &$residuals[$k];

                for ($i = 0; $i < $m; ++$i) {
                    $rk[$i] = $tk[$i] - $pk[$i];
                }

                unset($rk);
            }

            $lossChange = abs($prevLoss - $loss);

            $this->losses[$epoch] = $loss;

            // -------------------------------------------------------------
            // Step 3: Score the held-out set via argmax over current Ftest.
            // -------------------------------------------------------------
            if (isset($Ftest)) {
                $testPreds = $this->decodeScores($Ftest, $classes, $mTest);

                $score = $this->metric->score($testPreds, $testing->labels());

                $this->scores[$epoch] = $score;

                unset($testPreds);
            }

            // -------------------------------------------------------------
            // Step 4: Logging.
            // -------------------------------------------------------------
            if ($this->logger) {
                $dir     = $loss < $prevLoss ? '↓' : '↑';
                $message = "Epoch: $epoch, "
                    . "Multinomial Deviance: $loss, "
                    . "Loss Change: {$dir}{$lossChange}, "
                    . "{$this->metric}: " . ($score ?? 'N/A');

                $this->logger->info($message);
            }

            // -------------------------------------------------------------
            // Step 5: Early-stopping guards.
            // -------------------------------------------------------------
            if (is_nan($loss)) {
                if ($this->logger) {
                    $this->logger->warning('Numerical instability detected.');
                }

                break;
            }

            if (isset($score)) {
                if ($score >= $maxScore) {
                    break;
                }

                if ($score > $bestScore) {
                    $bestScore      = $score;
                    $bestEpoch      = $epoch;
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

            // -------------------------------------------------------------
            // Step 6: Fit K trees, one per class.
            //
            // For class k:
            //   a. Compute global Hessian Hₖ = mean(pₖ(1−pₖ)).
            //   b. Wrap samples + residuals[$k] in a pseudo-residual dataset.
            //   c. Draw weighted subsample and train a cloned booster.
            //   d. Store scale = η / Hₖ alongside the booster.
            //   e. Update F[k][i] += scale · pred[i] (training set).
            //   f. Mirror on held-out Ftest[k].
            //   g. Unset temporaries immediately.
            // -------------------------------------------------------------
            for ($k = 0; $k < $numClasses; ++$k) {

                // Per-class global Hessian with L2 regularisation.
                // Hₖ = (Σᵢ pₖ(1−pₖ) + λ) / m
                // The λ term prevents Hₖ from collapsing to 0 when class k's
                // probability mass vanishes (minority class starvation), which
                // would otherwise cause scale = η/Hₖ to explode and produce
                // oscillating, divergent F accumulators for minority classes.
                // Cache row references to avoid repeated two-level hash lookups.
                $pk = $probs[$k];

                $hessSum = $this->lambda;

                for ($i = 0; $i < $m; ++$i) {
                    $pi       = $pk[$i];
                    $hessSum += $pi * (1.0 - $pi);
                }

                $Hk = max($hessSum / $m, self::MIN_HESSIAN);

                $pseudoDataset = Labeled::quick($trainingSamples, $residuals[$k]);
                $subset        = $pseudoDataset->randomWeightedSubsetWithReplacement($p, $weights);

                $booster = clone $this->booster;
                $booster->train($subset);

                unset($pseudoDataset, $subset);

                $scale = $this->rate / $Hk;

                $this->boosters[$k][] = $booster;
                $this->scales[$k][]   = $scale;

                $treePreds = $booster->predict($training);
                $Fk        = &$F[$k];

                for ($i = 0; $i < $m; ++$i) {
                    $Fk[$i] += $scale * $treePreds[$i];
                }

                unset($Fk);

                unset($treePreds);

                if (isset($Ftest)) {
                    $testTreePreds = $booster->predict($testing);

                    for ($i = 0; $i < $mTest; ++$i) {
                        $Ftest[$k][$i] += $scale * $testTreePreds[$i];
                    }

                    unset($testTreePreds);
                }

                unset($booster);
            }

            // -------------------------------------------------------------
            // Step 7: Update sample weights.
            //
            // Per-sample weight = sum of |rᵢₖ| across all classes.
            // Concentrates future subsamples on the most uncertain samples.
            // Weights are zeroed first, then accumulated — no new allocation.
            // -------------------------------------------------------------
            // Seed from class 0 to avoid a separate zeroing pass.
            $rk0 = $residuals[0];

            for ($i = 0; $i < $m; ++$i) {
                $weights[$i] = abs($rk0[$i]);
            }

            for ($k = 1; $k < $numClasses; ++$k) {
                $rk = $residuals[$k];

                for ($i = 0; $i < $m; ++$i) {
                    $weights[$i] += abs($rk[$i]);
                }
            }

            $prevLoss = $loss;
        }
        // =================================================================
        // End multi-class boosting loop
        // =================================================================

        // Best-epoch restoration.
        if ($this->scores and end($this->scores) < $bestScore) {
            for ($k = 0; $k < $numClasses; ++$k) {
                $this->boosters[$k] = array_slice($this->boosters[$k], 0, $bestEpoch);
                $this->scales[$k]   = array_slice($this->scales[$k], 0, $bestEpoch);
            }

            if ($this->logger) {
                $this->logger->info("Model state restored to epoch $bestEpoch.");
            }
        }

        unset($F, $probs, $residuals, $targets, $classIndices, $Ftest);
    }

    // =========================================================================
    // Inference
    // =========================================================================

    /**
     * Make predictions from a dataset.
     *
     * Reconstructs the score accumulator(s) by replaying all stored boosters
     * with their Newton-scaled rates, then thresholds (binary) or argmax-es
     * (multi-class) to produce class labels.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<string>
     */
    public function predict(Dataset $dataset) : array
    {
        if (!isset($this->boosters, $this->scales, $this->classes, $this->featureCount)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->featureCount)->check();

        $n          = $dataset->numSamples();
        $numClasses = count($this->classes);

        if ($numClasses === 2) {
            return $this->predictBinary($dataset, $n);
        }

        return $this->predictMulticlass($dataset, $n, $numClasses);
    }

    /**
     * Reconstruct z for binary classification and threshold at 0.
     *
     * @param Dataset $dataset
     * @param int     $n
     * @return list<string>
     */
    protected function predictBinary(Dataset $dataset, int $n) : array
    {
        // Seed z at the training log-odds constant.
        /** @var float[] $z */
        $z = array_fill(0, $n, $this->z0 ?? 0.0);

        $epochs = count($this->boosters[0]);

        for ($t = 0; $t < $epochs; ++$t) {
            $preds = $this->boosters[0][$t]->predict($dataset);
            $s     = $this->scales[0][$t];

            for ($i = 0; $i < $n; ++$i) {
                $z[$i] += $s * $preds[$i];
            }

            unset($preds);
        }

        [$classA, $classB] = $this->classes;   // classA = 0, classB = 1

        $predictions = [];

        foreach ($z as $zVal) {
            $predictions[] = $zVal >= 0.0 ? $classB : $classA;
        }

        return $predictions;
    }

    /**
     * Reconstruct F[k] for multi-class classification and argmax.
     *
     * Softmax is not computed here: argmax over raw scores is equivalent to
     * argmax over softmax probabilities (monotone transformation), saving
     * K · n exponentiations.
     *
     * @param Dataset $dataset
     * @param int     $n
     * @param int     $numClasses
     * @return list<string>
     */
    protected function predictMulticlass(Dataset $dataset, int $n, int $numClasses) : array
    {
        // Seed accumulators at the training log-priors — must mirror the
        // initialisation used in trainMulticlass so the replay is exact.
        $f0 = $this->f0 ?? array_fill(0, $numClasses, 0.0);

        /** @var list<float[]> $F */
        $F = [];

        for ($k = 0; $k < $numClasses; ++$k) {
            $F[$k] = array_fill(0, $n, $f0[$k]);
        }

        $epochs = count($this->boosters[0]);

        for ($t = 0; $t < $epochs; ++$t) {
            for ($k = 0; $k < $numClasses; ++$k) {
                $preds = $this->boosters[$k][$t]->predict($dataset);
                $s     = $this->scales[$k][$t];

                for ($i = 0; $i < $n; ++$i) {
                    $F[$k][$i] += $s * $preds[$i];
                }

                unset($preds);
            }
        }

        return $this->decodeScores($F, $this->classes, $n);
    }

    /**
     * Estimate the joint probabilities for each possible outcome.
     *
     * Binary:     applies sigmoid to the reconstructed log-odds z.
     * Multi-class: applies numerically stable softmax to the reconstructed F matrix.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<array<string,float>>
     */
    public function proba(Dataset $dataset) : array
    {
        if (!isset($this->boosters, $this->scales, $this->classes, $this->featureCount)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, $this->featureCount)->check();

        $n          = $dataset->numSamples();
        $numClasses = count($this->classes);

        if ($numClasses === 2) {
            return $this->probaBinary($dataset, $n);
        }

        return $this->probaMulticlass($dataset, $n, $numClasses);
    }

    /**
     * Binary probability estimates via sigmoid on the log-odds accumulator.
     *
     * @param Dataset $dataset
     * @param int     $n
     * @return list<array<string,float>>
     */
    protected function probaBinary(Dataset $dataset, int $n) : array
    {
        /** @var float[] $z */
        $z = array_fill(0, $n, $this->z0 ?? 0.0);

        $epochs = count($this->boosters[0]);

        for ($t = 0; $t < $epochs; ++$t) {
            $preds = $this->boosters[0][$t]->predict($dataset);
            $s     = $this->scales[0][$t];

            for ($i = 0; $i < $n; ++$i) {
                $z[$i] += $s * $preds[$i];
            }

            unset($preds);
        }

        [$classA, $classB] = $this->classes;

        $probabilities = [];

        foreach ($z as $zVal) {
            $pPos = \Rubix\ML\sigmoid($zVal);

            $probabilities[] = [
                $classA => 1.0 - $pPos,
                $classB => $pPos,
            ];
        }

        return $probabilities;
    }

    /**
     * Multi-class probability estimates via numerically stable softmax.
     *
     * @param Dataset $dataset
     * @param int     $n
     * @param int     $numClasses
     * @return list<array<string,float>>
     */
    protected function probaMulticlass(Dataset $dataset, int $n, int $numClasses) : array
    {
        // Seed accumulators at the training log-priors — must mirror the
        // initialisation used in trainMulticlass so the replay is exact.
        $f0 = $this->f0 ?? array_fill(0, $numClasses, 0.0);

        /** @var list<float[]> $F */
        $F = [];

        for ($k = 0; $k < $numClasses; ++$k) {
            $F[$k] = array_fill(0, $n, $f0[$k]);
        }

        $epochs = count($this->boosters[0]);

        for ($t = 0; $t < $epochs; ++$t) {
            for ($k = 0; $k < $numClasses; ++$k) {
                $preds = $this->boosters[$k][$t]->predict($dataset);
                $s     = $this->scales[$k][$t];

                for ($i = 0; $i < $n; ++$i) {
                    $F[$k][$i] += $s * $preds[$i];
                }

                unset($preds);
            }
        }

        $probabilities = [];

        for ($i = 0; $i < $n; ++$i) {
            // Per-sample max for numerical stability.
            $maxF = $F[0][$i];

            for ($k = 1; $k < $numClasses; ++$k) {
                if ($F[$k][$i] > $maxF) {
                    $maxF = $F[$k][$i];
                }
            }

            $expSum = 0.0;

            /** @var float[] $expVals */
            $expVals = array_fill(0, $numClasses, 0.0);

            for ($k = 0; $k < $numClasses; ++$k) {
                $e          = exp($F[$k][$i] - $maxF);
                $expVals[$k] = $e;
                $expSum     += $e;
            }

            $dist = [];

            for ($k = 0; $k < $numClasses; ++$k) {
                $dist[$this->classes[$k]] = $expVals[$k] / $expSum;
            }

            $probabilities[] = $dist;

            unset($expVals);
        }

        unset($F);

        return $probabilities;
    }

    /**
     * Return the importance scores of each feature column of the training set.
     *
     * Averages importance over every booster across every class slot.
     *
     * @throws RuntimeException
     * @return float[]
     */
    public function featureImportances() : array
    {
        if (!isset($this->boosters, $this->featureCount)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        /** @var float[] $importances */
        $importances = array_fill(0, $this->featureCount, 0.0);
        $numBoosters = 0;

        foreach ($this->boosters as $classBoosters) {
            foreach ($classBoosters as $tree) {
                $scores = $tree->featureImportances();

                foreach ($scores as $column => $score) {
                    $importances[$column] += $score;
                }

                ++$numBoosters;
            }
        }

        if ($numBoosters > 0) {
            foreach ($importances as &$importance) {
                $importance /= $numBoosters;
            }
        }

        return $importances;
    }

    /**
     * Decode a per-class score matrix into a flat list of predicted class labels
     * by taking the argmax over raw F scores (softmax is a monotone transform,
     * so argmax(F) == argmax(softmax(F)), saving K·n exp() calls).
     *
     * @param list<float[]> $F       Per-class score matrix $F[$k][$i].
     * @param list<string>  $classes Ordered class label list.
     * @param int           $n       Number of samples.
     * @return list<string>
     */
    protected function decodeScores(array $F, array $classes, int $n) : array
    {
        $numClasses  = count($classes);
        $predictions = [];

        for ($i = 0; $i < $n; ++$i) {
            $bestK = 0;
            $bestF = $F[0][$i];

            for ($k = 1; $k < $numClasses; ++$k) {
                if ($F[$k][$i] > $bestF) {
                    $bestF = $F[$k][$i];
                    $bestK = $k;
                }
            }

            $predictions[] = $classes[$bestK];
        }

        return $predictions;
    }

    /**
     * Return the data used to serialise the object.
     * Training history (losses, scores) and the logger are excluded — they are
     * ephemeral and need not survive serialisation.
     *
     * @return mixed[]
     */
    public function __serialize() : array
    {
        $properties = get_object_vars($this);

        unset($properties['losses'], $properties['scores'], $properties['logger']);

        return $properties;
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Gradient Boost Classifier (' . Params::stringify($this->params()) . ')';
    }
}
