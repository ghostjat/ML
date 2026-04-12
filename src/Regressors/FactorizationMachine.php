<?php

namespace Rubix\ML\Regressors;

use Rubix\ML\Learner;
use Rubix\ML\Online;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
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
use function sqrt;
use function array_fill;
use function is_null;

use const Rubix\ML\EPSILON;

/**
 * Factorization Machine Regressor
 *
 * A second-order feature interaction model trained with mini-batch SGD using the
 * mean-squared error loss.  The interaction term is computed in O(kn) per sample
 * using the FM identity: Σ_{i<j} <v_i,v_j> x_i x_j = 0.5 Σ_f [(Σ_i v_{if} x_i)² - Σ_i v²_{if} x²_i].
 *
 * References:
 * [1] S. Rendle. (2010). Factorization Machines.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      The Rubix ML Community
 */
class FactorizationMachine implements Estimator, Learner, Online, Persistable
{
    use AutotrackRevisions;

    /**
     * Number of latent factors per feature.
     *
     * @var int
     */
    protected int $factors;

    /**
     * Initial SGD learning rate.
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
     * Global bias term.
     *
     * @var float|null
     */
    protected ?float $bias = null;

    /**
     * First-order weight vector: float[p].
     *
     * @var float[]|null
     */
    protected ?array $weights = null;

    /**
     * Factor matrix V: float[p][k].
     *
     * @var float[][]|null
     */
    protected ?array $V = null;

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
        return EstimatorType::regressor();
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
        return isset($this->weights) and isset($this->bias);
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
        $p     = $dataset->numFeatures();
        $k     = $this->factors;
        $scale = 1.0 / sqrt($k);

        $this->bias    = 0.0;
        $this->weights = array_fill(0, $p, 0.0);

        $V = [];
        for ($j = 0; $j < $p; ++$j) {
            $row = [];
            for ($f = 0; $f < $k; ++$f) {
                $row[] = (lcg_value() * 2.0 - 1.0) * $scale;
            }
            $V[] = $row;
        }
        $this->V = $V;

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
            new DatasetHasDimensionality($dataset, count($this->weights)),
        ])->check();

        /** @var \Rubix\ML\Datasets\Labeled $dataset */
        $samples = $dataset->samples();
        $labels  = $dataset->labels();
        $n       = count($samples);
        $k       = $this->factors;
        $alpha   = $this->rate;
        $lambda  = $this->l2Penalty;

        for ($epoch = 0; $epoch < $this->epochs; ++$epoch) {
            $indices = range(0, $n - 1);
            shuffle($indices);

            $bStart = 0;
            while ($bStart < $n) {
                $bEnd = min($bStart + $this->batchSize, $n);

                for ($bi = $bStart; $bi < $bEnd; ++$bi) {
                    $idx  = $indices[$bi];
                    $x    = $samples[$idx];
                    $y    = (float) $labels[$idx];
                    $p    = count($x);

                    $yHat  = $this->fmScore($x);
                    $delta = $yHat - $y;   // dL/dy_hat for MSE = (y_hat - y)

                    // Gradient for bias.
                    $this->bias -= $alpha * $delta;

                    // Precompute Σ_j v_{jf} x_j for each factor.
                    $V     = $this->V;
                    $sumVX = array_fill(0, $k, 0.0);
                    for ($j = 0; $j < $p; ++$j) {
                        $xj = $x[$j];
                        for ($f = 0; $f < $k; ++$f) {
                            $sumVX[$f] += $V[$j][$f] * $xj;
                        }
                    }

                    for ($j = 0; $j < $p; ++$j) {
                        $xj = $x[$j];

                        // First-order weight gradient.
                        $this->weights[$j] -= $alpha * ($delta * $xj + $lambda * $this->weights[$j]);

                        // Factor gradient.
                        for ($f = 0; $f < $k; ++$f) {
                            $grad = $delta * $xj * ($sumVX[$f] - $V[$j][$f] * $xj);
                            $this->V[$j][$f] -= $alpha * ($grad + $lambda * $V[$j][$f]);
                        }
                    }
                }

                $bStart = $bEnd;
            }
        }
    }

    /**
     * Make continuous predictions for a dataset.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<int|float>
     */
    public function predict(Dataset $dataset) : array
    {
        if (!isset($this->weights) or is_null($this->bias)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count($this->weights))->check();

        $predictions = [];

        foreach ($dataset->samples() as $x) {
            $predictions[] = $this->fmScore($x);
        }

        return $predictions;
    }

    /**
     * Compute the FM output score for a single sample.
     *
     * @param list<int|float> $x
     * @return float
     */
    protected function fmScore(array $x) : float
    {
        $score  = $this->bias;
        $w      = $this->weights;
        $V      = $this->V;
        $k      = $this->factors;
        $p      = count($x);

        // First-order.
        for ($j = 0; $j < $p; ++$j) {
            $score += $w[$j] * $x[$j];
        }

        // Second-order O(kp).
        $inter = 0.0;
        for ($f = 0; $f < $k; ++$f) {
            $sumVX  = 0.0;
            $sumVX2 = 0.0;
            for ($j = 0; $j < $p; ++$j) {
                $vx     = $V[$j][$f] * $x[$j];
                $sumVX += $vx;
                $sumVX2 += $vx * $vx;
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
        return 'Factorization Machine (Regressor) (' . Params::stringify($this->params()) . ')';
    }
}
