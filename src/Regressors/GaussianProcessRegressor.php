<?php

namespace Rubix\ML\Regressors;

use Tensor\Matrix;
use Tensor\Vector;
use Rubix\ML\Learner;
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
use function exp;
use function sqrt;
use function max;
use function is_null;

use const Rubix\ML\EPSILON;

/**
 * Gaussian Process Regressor
 *
 * A non-parametric Bayesian regressor that models the posterior distribution over
 * functions using a Gaussian process prior. The RBF (squared-exponential) kernel
 * measures similarity between samples. Training is O(n³) due to the covariance
 * matrix inversion; suitable for datasets up to ~5 000 samples.
 *
 * References:
 * [1] C. E. Rasmussen & C. K. I. Williams. (2006). Gaussian Processes for Machine Learning.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      The Rubix ML Community
 */
class GaussianProcessRegressor implements Estimator, Learner, Persistable
{
    use AutotrackRevisions;

    /**
     * Length-scale of the RBF kernel.  Larger values give smoother functions.
     *
     * @var float
     */
    protected float $lengthScale;

    /**
     * Signal amplitude (output scale) of the RBF kernel.
     *
     * @var float
     */
    protected float $amplitude;

    /**
     * Independent Gaussian noise variance added to the diagonal of the kernel matrix.
     *
     * @var float
     */
    protected float $noiseVariance;

    /**
     * The training samples stored for computing the test kernel.
     *
     * @var list<list<int|float>>|null
     */
    protected ?array $trainSamples = null;

    /**
     * α = K^{-1} y – the dual coefficients used for mean prediction.
     *
     * @var float[]|null
     */
    protected ?array $alpha = null;

    /**
     * Inverted training kernel matrix K^{-1} used for variance prediction.
     *
     * @var Matrix|null
     */
    protected ?Matrix $kInv = null;

    /**
     * @param float $lengthScale
     * @param float $amplitude
     * @param float $noiseVariance
     * @throws InvalidArgumentException
     */
    public function __construct(
        float $lengthScale = 1.0,
        float $amplitude = 1.0,
        float $noiseVariance = 1e-4
    ) {
        if ($lengthScale <= 0.0) {
            throw new InvalidArgumentException('Length scale must be greater'
                . " than 0, $lengthScale given.");
        }

        if ($amplitude <= 0.0) {
            throw new InvalidArgumentException('Amplitude must be greater'
                . " than 0, $amplitude given.");
        }

        if ($noiseVariance < 0.0) {
            throw new InvalidArgumentException('Noise variance must be'
                . " greater than or equal to 0, $noiseVariance given.");
        }

        $this->lengthScale   = $lengthScale;
        $this->amplitude     = $amplitude;
        $this->noiseVariance = $noiseVariance;
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
            'length scale'   => $this->lengthScale,
            'amplitude'      => $this->amplitude,
            'noise variance' => $this->noiseVariance,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return isset($this->alpha) and isset($this->kInv);
    }

    /**
     * Train the GP by computing and inverting the kernel matrix.
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

        // Build K(X_train, X_train) + σ_n² I
        $kData = $this->kernelMatrix($samples, $samples);
        for ($i = 0; $i < $n; ++$i) {
            $kData[$i][$i] += $this->noiseVariance;
        }

        $k = Matrix::quick($kData);

        // α = K^{-1} y
        $y = [];
        for ($i = 0; $i < $n; ++$i) {
            $y[$i] = (float) $labels[$i];
        }
        $yVec = Vector::quick($y);

        $kInv = $k->inverse();

        $this->alpha       = $kInv->dot($yVec)->asArray();
        $this->kInv        = $kInv;
        $this->trainSamples = $samples;
    }

    /**
     * Predict the posterior mean for each sample.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<int|float>
     */
    public function predict(Dataset $dataset) : array
    {
        if (!isset($this->alpha) or is_null($this->trainSamples)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count($this->trainSamples[0]))->check();

        $testSamples = $dataset->samples();
        $kStar = $this->kernelMatrix($testSamples, $this->trainSamples);

        $predictions = [];
        $alpha = $this->alpha;

        foreach ($kStar as $row) {
            $mu = 0.0;
            foreach ($row as $j => $kij) {
                $mu += $kij * $alpha[$j];
            }
            $predictions[] = $mu;
        }

        return $predictions;
    }

    /**
     * Return the posterior mean and standard deviation for each sample.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return array{mean: float[], std: float[]}
     */
    public function predictWithVariance(Dataset $dataset) : array
    {
        if (!isset($this->kInv) or is_null($this->trainSamples)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count($this->trainSamples[0]))->check();

        $testSamples = $dataset->samples();
        $m = count($testSamples);

        $kStarData = $this->kernelMatrix($testSamples, $this->trainSamples);
        $kStar = Matrix::quick($kStarData);

        // Posterior means: K* α
        $alpha = $this->alpha;
        $means = [];
        foreach ($kStarData as $row) {
            $mu = 0.0;
            foreach ($row as $j => $kij) {
                $mu += $kij * $alpha[$j];
            }
            $means[] = $mu;
        }

        // Posterior variances: k** - diag(K* K^{-1} K*^T)
        $amp2 = $this->amplitude * $this->amplitude;

        // K* K^{-1}: (m × n) × (n × n) = (m × n)
        $kStarKInv = $kStar->matmul($this->kInv)->asArray();

        $stds = [];
        foreach ($kStarKInv as $i => $row) {
            $vDiag = 0.0;
            $kStarRow = $kStarData[$i];
            foreach ($row as $j => $v) {
                $vDiag += $v * $kStarRow[$j];
            }
            $variance = max(0.0, $amp2 - $vDiag);
            $stds[]   = sqrt($variance);
        }

        return ['mean' => $means, 'std' => $stds];
    }

    /**
     * Compute the RBF kernel matrix between two sets of samples.
     *
     * k(x, x') = amplitude² * exp(-||x - x'||² / (2 * l²))
     *
     * @param list<list<int|float>> $a
     * @param list<list<int|float>> $b
     * @return array<float[]>
     */
    protected function kernelMatrix(array $a, array $b) : array
    {
        $inv2l2 = 1.0 / (2.0 * $this->lengthScale * $this->lengthScale);
        $amp2   = $this->amplitude * $this->amplitude;
        $matrix = [];

        foreach ($a as $ai) {
            $row = [];

            foreach ($b as $bi) {
                $ssd = 0.0;
                foreach ($ai as $d => $v) {
                    $diff = $v - $bi[$d];
                    $ssd += $diff * $diff;
                }
                $row[] = $amp2 * exp(-$ssd * $inv2l2);
            }

            $matrix[] = $row;
        }

        return $matrix;
    }

    /**
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Gaussian Process Regressor (' . Params::stringify($this->params()) . ')';
    }
}
