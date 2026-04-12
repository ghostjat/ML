<?php

namespace Rubix\ML\Regressors;

use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\RanksFeatures;
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
use function abs;
use function max;
use function array_fill;
use function array_map;
use function is_null;

use const Rubix\ML\EPSILON;

/**
 * Elastic Net
 *
 * A linear regressor trained via coordinate descent that blends L1 (Lasso) and L2
 * (Ridge) regularization in a single penalty term. The l1Ratio controls the mixing:
 * 0 = pure Ridge, 1 = pure Lasso, values in between give the elastic net effect.
 * Coordinate descent maintains a residual vector for O(np) updates per epoch.
 *
 * References:
 * [1] H. Zou et al. (2005). Regularization and Variable Selection via the Elastic Net.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      The Rubix ML Community
 */
class ElasticNet implements Estimator, Learner, RanksFeatures, Persistable
{
    use AutotrackRevisions;

    /**
     * The combined regularization strength (alpha >= 0).
     *
     * @var float
     */
    protected float $alpha;

    /**
     * Mixing ratio between L1 and L2 (0 = Ridge, 1 = Lasso).
     *
     * @var float
     */
    protected float $l1Ratio;

    /**
     * Maximum coordinate descent iterations per training call.
     *
     * @var int
     */
    protected int $epochs;

    /**
     * Convergence tolerance on the maximum weight change per epoch.
     *
     * @var float
     */
    protected float $tol;

    /**
     * The bias (intercept) of the learned model.
     *
     * @var float|null
     */
    protected ?float $bias = null;

    /**
     * The learned weight coefficients indexed by feature column.
     *
     * @var float[]|null
     */
    protected ?array $weights = null;

    /**
     * @param float $alpha
     * @param float $l1Ratio
     * @param int $epochs
     * @param float $tol
     * @throws InvalidArgumentException
     */
    public function __construct(
        float $alpha = 1.0,
        float $l1Ratio = 0.5,
        int $epochs = 1000,
        float $tol = 1e-4
    ) {
        if ($alpha < 0.0) {
            throw new InvalidArgumentException('Alpha must be greater than'
                . " or equal to 0, $alpha given.");
        }

        if ($l1Ratio < 0.0 or $l1Ratio > 1.0) {
            throw new InvalidArgumentException('L1 ratio must be between'
                . " 0 and 1, $l1Ratio given.");
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Number of epochs must be'
                . " greater than 0, $epochs given.");
        }

        if ($tol < 0.0) {
            throw new InvalidArgumentException('Tolerance must be greater'
                . " than or equal to 0, $tol given.");
        }

        $this->alpha   = $alpha;
        $this->l1Ratio = $l1Ratio;
        $this->epochs  = $epochs;
        $this->tol     = $tol;
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
            'alpha'    => $this->alpha,
            'l1 ratio' => $this->l1Ratio,
            'epochs'   => $this->epochs,
            'tol'      => $this->tol,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return isset($this->weights) and isset($this->bias);
    }

    /**
     * Return the learned weight coefficients or null if not trained.
     *
     * @return float[]|null
     */
    public function coefficients() : ?array
    {
        return $this->weights;
    }

    /**
     * Return the learned bias or null if not trained.
     *
     * @return float|null
     */
    public function bias() : ?float
    {
        return $this->bias;
    }

    /**
     * Train the learner with a labeled dataset.
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
        $samples = $dataset->samples();
        $labels  = $dataset->labels();

        $n = count($samples);
        $p = count($samples[0]);

        $l1 = $this->alpha * $this->l1Ratio;
        $l2 = $this->alpha * (1.0 - $this->l1Ratio);

        // Zero-initialised weights and bias.
        $w = array_fill(0, $p, 0.0);
        $b = 0.0;

        // Residuals r[i] = y[i] - X[i]·w - b  (with w=0, b=0: r=y initially).
        $r = [];
        for ($i = 0; $i < $n; ++$i) {
            $r[$i] = (float) $labels[$i];
        }

        // Precompute per-feature column norm² / n  (invariant across epochs).
        $normSq = array_fill(0, $p, 0.0);
        for ($j = 0; $j < $p; ++$j) {
            $sq = 0.0;
            for ($i = 0; $i < $n; ++$i) {
                $v = $samples[$i][$j];
                $sq += $v * $v;
            }
            $normSq[$j] = $sq / $n;
        }

        for ($epoch = 0; $epoch < $this->epochs; ++$epoch) {
            $maxDelta = 0.0;

            // --- Update intercept: b_new = b + mean(r) ---
            $rSum = 0.0;
            for ($i = 0; $i < $n; ++$i) {
                $rSum += $r[$i];
            }
            $bDelta = $rSum / $n;

            if ($bDelta !== 0.0) {
                for ($i = 0; $i < $n; ++$i) {
                    $r[$i] -= $bDelta;
                }
                $absDelta = $bDelta < 0.0 ? -$bDelta : $bDelta;
                if ($absDelta > $maxDelta) {
                    $maxDelta = $absDelta;
                }
                $b += $bDelta;
            }

            // --- Coordinate descent over features ---
            for ($j = 0; $j < $p; ++$j) {
                if ($normSq[$j] < EPSILON) {
                    continue;
                }

                // rho_j = X_j · r / n  +  w_j  (add back this feature's contribution)
                $rho = 0.0;
                for ($i = 0; $i < $n; ++$i) {
                    $rho += $samples[$i][$j] * $r[$i];
                }
                $rho = $rho / $n + $w[$j];

                $wNew = $this->softThreshold($rho, $l1) / ($normSq[$j] + $l2);

                $wDelta = $wNew - $w[$j];

                if ($wDelta !== 0.0) {
                    // Update residuals in-place: avoids recomputing predictions.
                    for ($i = 0; $i < $n; ++$i) {
                        $r[$i] -= $wDelta * $samples[$i][$j];
                    }
                    $absDelta = $wDelta < 0.0 ? -$wDelta : $wDelta;
                    if ($absDelta > $maxDelta) {
                        $maxDelta = $absDelta;
                    }
                    $w[$j] = $wNew;
                }
            }

            if ($maxDelta < $this->tol) {
                break;
            }
        }

        $this->bias    = $b;
        $this->weights = $w;
    }

    /**
     * Make continuous predictions on a dataset.
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

        foreach ($dataset->samples() as $sample) {
            $output = $this->bias;

            foreach ($this->weights as $j => $wj) {
                $output += $wj * $sample[$j];
            }

            $predictions[] = $output;
        }

        return $predictions;
    }

    /**
     * Return the absolute values of the weight coefficients as feature importances.
     *
     * @throws RuntimeException
     * @return float[]
     */
    public function featureImportances() : array
    {
        if (!isset($this->weights)) {
            throw new RuntimeException('Learner has not been trained.');
        }

        return array_map('abs', $this->weights);
    }

    /**
     * Soft-thresholding operator: sign(z) * max(|z| - gamma, 0).
     *
     * @param float $z
     * @param float $gamma
     * @return float
     */
    protected function softThreshold(float $z, float $gamma) : float
    {
        if ($z > $gamma) {
            return $z - $gamma;
        }

        if ($z < -$gamma) {
            return $z + $gamma;
        }

        return 0.0;
    }

    /**
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Elastic Net (' . Params::stringify($this->params()) . ')';
    }
}
