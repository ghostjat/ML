<?php

declare(strict_types=1);

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
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\DatasetIsLabeled;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function count;
use function max;
use function min;
use function abs;
use function array_map;
use function array_slice;
use function array_values;
use function array_sum;
use function array_fill;
use function array_shift;
use function array_unshift;

/**
 * ARIMA
 *
 * AutoRegressive Integrated Moving Average (ARIMA) is a classic statistical
 * model for time-series analysis and forecasting. The ARIMA(p, d, q) model
 * combines three components:
 *
 * - AR(p): uses p lagged observations to model autocorrelation
 * - I(d):  applies d-th order differencing to induce stationarity
 * - MA(q): uses q lagged forecast residuals to model shock propagation
 *
 * Parameters are estimated via the Hannan-Rissanen two-stage algorithm,
 * which first fits a long AR model to approximate the innovations and then
 * performs OLS on the combined AR + MA feature matrix.
 *
 * **Training format**: Pass a `Labeled` dataset whose *labels* form the
 * chronologically ordered time series. Feature columns are ignored (timestamps
 * or any placeholder values are fine).
 *
 * **Prediction format**: Each sample must contain exactly `windowSize()`
 * = (d + p) consecutive original-scale observations ending immediately
 * before the target step: [y(t-d-p), …, y(t-1)] → ŷ(t).
 *
 * **Forecasting**: Call `forecast(int $steps)` to generate future values
 * directly from the end of the training series.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      The Rubix ML Community
 */
class Arima implements Estimator, Learner, Persistable
{
    use AutotrackRevisions;

    /**
     * The autoregressive order (number of lagged observations).
     *
     * @var int<0,max>
     */
    protected int $p;

    /**
     * The degree of differencing applied to achieve stationarity.
     *
     * @var int<0,max>
     */
    protected int $d;

    /**
     * The moving-average order (number of lagged residuals).
     *
     * @var int<0,max>
     */
    protected int $q;

    /**
     * Fitted AR coefficients φ₁, …, φₚ.
     *
     * @var float[]|null
     */
    protected ?array $phi = null;

    /**
     * Fitted MA coefficients θ₁, …, θq.
     *
     * @var float[]|null
     */
    protected ?array $theta = null;

    /**
     * Mean of the differenced (and centered) series.
     *
     * @var float|null
     */
    protected ?float $mu = null;

    /**
     * Rolling buffer of the last (d + max(p,1)) original-scale values,
     * used as the seed for multi-step forecasting.
     *
     * @var float[]|null
     */
    protected ?array $buffer = null;

    /**
     * Rolling residuals (length q) at the end of training,
     * used to seed MA terms during the first q forecast steps.
     *
     * @var float[]|null
     */
    protected ?array $errors = null;

    /**
     * @param int $p AR order (≥ 0)
     * @param int $d Differencing order (≥ 0)
     * @param int $q MA order (≥ 0)
     * @throws InvalidArgumentException
     */
    public function __construct(int $p = 1, int $d = 1, int $q = 0)
    {
        if ($p < 0) {
            throw new InvalidArgumentException("AR order p must be"
                . " non-negative, $p given.");
        }

        if ($d < 0) {
            throw new InvalidArgumentException("Differencing order d must be"
                . " non-negative, $d given.");
        }

        if ($q < 0) {
            throw new InvalidArgumentException("MA order q must be"
                . " non-negative, $q given.");
        }

        $this->p = $p;
        $this->d = $d;
        $this->q = $q;
    }

    /**
     * Return the estimator type.
     *
     * @return EstimatorType
     */
    public function type() : EstimatorType
    {
        return EstimatorType::regressor();
    }

    /**
     * Return the data types this estimator is compatible with.
     *
     * @return list<DataType>
     */
    public function compatibility() : array
    {
        return [DataType::continuous()];
    }

    /**
     * Return the hyper-parameter settings.
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'p' => $this->p,
            'd' => $this->d,
            'q' => $this->q,
        ];
    }

    /**
     * Has the model been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return $this->phi !== null;
    }

    /**
     * Return the number of features expected per sample in `predict()`.
     * Each sample should supply the last windowSize() = (d + p)
     * original-scale observations before the target step.
     *
     * @return int
     */
    public function windowSize() : int
    {
        return $this->d + $this->p;
    }

    /**
     * Fit the ARIMA model to a time series.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset A Labeled dataset whose
     *   labels are the chronologically ordered time series values.
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        SpecificationChain::with([
            new DatasetIsLabeled($dataset),
            new DatasetIsNotEmpty($dataset),
        ])->check();

        /** @var Labeled $dataset */
        $series = array_values(array_map('floatval', $dataset->labels()));
        $n = count($series);

        // Long-AR order for Hannan-Rissanen stage 1
        $longAR = max(25, 2 * max($this->p, $this->q) + 5);
        $minRequired = $this->d + $longAR + max($this->p, $this->q) + 10;

        if ($n < $minRequired) {
            throw new InvalidArgumentException(
                "Time series needs at least {$minRequired} observations,"
                . " {$n} given."
            );
        }

        // Step 1: apply d-order differencing
        $differenced = $this->applyDifference($series, $this->d);

        // Step 2: center the differenced series
        $this->mu = array_sum($differenced) / count($differenced);
        $centered  = array_map(fn ($v) => $v - $this->mu, $differenced);

        // Step 3: fit ARMA(p, q) via Hannan-Rissanen
        $this->fitArma($centered, $longAR);

        // Store the last (d + max(p, 1)) original values as the forecast seed
        $bufLen = $this->d + max($this->p, 1);
        $this->buffer = array_slice($series, -$bufLen);

        // Compute trailing residuals to initialise MA terms during forecast
        $this->errors = $this->computeLastResiduals($centered);
    }

    /**
     * Predict the next value for each context window in the dataset.
     *
     * Each sample must contain exactly `windowSize()` = (d + p)
     * original-scale values in chronological order.
     *
     * @param Dataset $dataset
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return list<float>
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->trained()) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $ws = $this->windowSize();

        if ($ws > 0) {
            DatasetHasDimensionality::with($dataset, $ws)->check();
        }

        $predictions = [];

        foreach ($dataset->samples() as $sample) {
            $context = array_map('floatval', $sample);
            $predictions[] = $this->predictFromContext($context, []);
        }

        return $predictions;
    }

    /**
     * Generate future values from the end of the training series.
     *
     * @param int $steps Number of steps ahead to forecast (≥ 1)
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return float[]
     */
    public function forecast(int $steps = 1) : array
    {
        if (!$this->trained()) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        if ($steps < 1) {
            throw new InvalidArgumentException(
                "Steps must be at least 1, {$steps} given."
            );
        }

        $forecasts = [];
        $buffer    = $this->buffer;
        $errors    = $this->errors ?? [];

        for ($step = 0; $step < $steps; $step++) {
            $prediction = $this->predictFromContext($buffer, $errors);
            $forecasts[] = $prediction;

            // Append new prediction, drop oldest original value
            $buffer[] = $prediction;

            $maxBuf = $this->d + max($this->p, 1);
            while (count($buffer) > $maxBuf) {
                array_shift($buffer);
            }

            // Future shock is unknown → assume 0 (minimum MSE point forecast)
            array_unshift($errors, 0.0);
            if (count($errors) > $this->q) {
                array_pop($errors);
            }
        }

        return $forecasts;
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    /**
     * Produce a one-step prediction from an original-scale context window
     * and optional known residuals.
     *
     * @param float[] $context  Last (d + p) original-scale values
     * @param float[] $residuals Last q residuals (empty = assume 0)
     * @return float
     */
    protected function predictFromContext(array $context, array $residuals) : float
    {
        // Difference the context
        if ($this->d > 0) {
            $diffed   = $this->applyDifference(array_values($context), $this->d);
            $centered = array_map(fn ($v) => $v - $this->mu, $diffed);
        } else {
            $centered = array_map(fn ($v) => $v - $this->mu, array_values($context));
        }

        $n = count($centered);

        // AR contribution
        $pred = 0.0;

        foreach ($this->phi as $i => $phi) {
            $idx = $n - 1 - $i;
            if ($idx >= 0) {
                $pred += $phi * $centered[$idx];
            }
        }

        // MA contribution (missing residuals → 0)
        foreach ($this->theta as $j => $theta) {
            $pred += $theta * ($residuals[$j] ?? 0.0);
        }

        // Re-add mean of differenced series
        $diffPred = $pred + $this->mu;

        // Invert differencing
        if ($this->d > 0) {
            return $this->integrateDifference($diffPred, array_values($context), $this->d);
        }

        return $diffPred;
    }

    /**
     * Apply d-order differencing to a series.
     *
     * @param float[] $series
     * @param int     $d
     * @return float[]
     */
    protected function applyDifference(array $series, int $d) : array
    {
        $result = $series;

        for ($i = 0; $i < $d; $i++) {
            $next = [];

            for ($j = 1, $len = count($result); $j < $len; $j++) {
                $next[] = $result[$j] - $result[$j - 1];
            }

            $result = $next;
        }

        return $result;
    }

    /**
     * Invert one step of d-th order differencing using the binomial formula.
     *
     * y(t) = w + Σᵢ₌₁ᵈ (-1)^(i+1) · C(d,i) · y(t-i)
     *
     * @param float   $w        d-th order differenced prediction
     * @param float[] $context  Original-scale context [y(t-n), …, y(t-1)]
     * @param int     $d
     * @return float
     */
    protected function integrateDifference(float $w, array $context, int $d) : float
    {
        $result = $w;
        $n      = count($context);

        for ($i = 1; $i <= $d; $i++) {
            $sign   = (($i % 2) === 1) ? 1.0 : -1.0;
            $binom  = $this->binomialCoefficient($d, $i);
            $idx    = $n - $i;   // y(t-i) sits at position [n-i]

            if ($idx >= 0) {
                $result += $sign * $binom * $context[$idx];
            }
        }

        return $result;
    }

    /**
     * Compute the binomial coefficient C(n, k).
     */
    protected function binomialCoefficient(int $n, int $k) : float
    {
        if ($k === 0 || $k === $n) {
            return 1.0;
        }

        $k      = min($k, $n - $k);
        $result = 1.0;

        for ($i = 0; $i < $k; $i++) {
            $result *= ($n - $i) / ($i + 1);
        }

        return $result;
    }

    /**
     * Fit ARMA(p, q) via the two-stage Hannan-Rissanen algorithm.
     *
     * Stage 1: fit a long AR(m) model to approximate the innovations.
     * Stage 2: OLS on [y(t-1),…,y(t-p), ε̂(t-1),…,ε̂(t-q)] → y(t).
     *
     * @param float[] $centered Centered differenced series
     * @param int     $longAR   Long-AR order for stage 1
     */
    protected function fitArma(array $centered, int $longAR) : void
    {
        $n = count($centered);

        // Stage 1 – long AR model
        $longARCoeffs = $this->fitLongAR($centered, min($longAR, (int) ($n / 3)));
        $m            = count($longARCoeffs);

        // Stage 1 residuals (innovations)
        $innovations = array_fill(0, $n, 0.0);

        for ($t = $m; $t < $n; $t++) {
            $pred = 0.0;

            for ($i = 0; $i < $m; $i++) {
                $pred += $longARCoeffs[$i] * $centered[$t - 1 - $i];
            }

            $innovations[$t] = $centered[$t] - $pred;
        }

        // Trivial case: no AR or MA terms
        if ($this->p === 0 && $this->q === 0) {
            $this->phi   = [];
            $this->theta = [];

            return;
        }

        // Stage 2 – build design matrix
        $startIdx = max($this->p, $this->q, $m);
        $X        = [];
        $y        = [];

        for ($t = $startIdx; $t < $n; $t++) {
            $row = [];

            for ($i = 1; $i <= $this->p; $i++) {
                $row[] = $centered[$t - $i];
            }

            for ($j = 1; $j <= $this->q; $j++) {
                $row[] = $innovations[$t - $j];
            }

            if (!empty($row)) {
                $X[] = $row;
                $y[] = $centered[$t];
            }
        }

        if (empty($X)) {
            $this->phi   = array_fill(0, $this->p, 0.0);
            $this->theta = array_fill(0, $this->q, 0.0);

            return;
        }

        $coeffs      = $this->solveOLS($X, $y);
        $this->phi   = array_slice($coeffs, 0, $this->p);
        $this->theta = array_slice($coeffs, $this->p, $this->q);
    }

    /**
     * Fit a long AR model using the Yule-Walker equations.
     *
     * @param float[] $series
     * @param int     $order
     * @return float[]
     */
    protected function fitLongAR(array $series, int $order) : array
    {
        if ($order === 0) {
            return [];
        }

        $n    = count($series);
        $mean = array_sum($series) / $n;

        // Sample variance
        $var = 0.0;

        foreach ($series as $v) {
            $var += ($v - $mean) ** 2;
        }

        if ($var < 1e-14) {
            return array_fill(0, $order, 0.0);
        }

        // Autocorrelations r[0], r[1], …, r[order]
        $r = [];

        for ($k = 0; $k <= $order; $k++) {
            $cov = 0.0;

            for ($t = $k; $t < $n; $t++) {
                $cov += ($series[$t] - $mean) * ($series[$t - $k] - $mean);
            }

            $r[$k] = $cov / $var;
        }

        // Yule-Walker system: R * φ = r[1..order]
        $R   = [];
        $rhs = [];

        for ($i = 0; $i < $order; $i++) {
            $row = [];

            for ($j = 0; $j < $order; $j++) {
                $row[] = $r[abs($i - $j)];
            }

            $R[]   = $row;
            $rhs[] = $r[$i + 1];
        }

        return $this->solveLinearSystem($R, $rhs, 1e-8);
    }

    /**
     * Solve the OLS normal equations β = (X^T X + λI)^{-1} X^T y.
     *
     * @param float[][] $X
     * @param float[]   $y
     * @param float     $lambda Ridge regularisation for numerical stability
     * @return float[]
     */
    protected function solveOLS(array $X, array $y, float $lambda = 1e-8) : array
    {
        $k = count($X[0]);

        if ($k === 0) {
            return [];
        }

        // Fast path for scalar case
        if ($k === 1) {
            $xtx = 0.0;
            $xty = 0.0;

            foreach ($X as $i => $row) {
                $xtx += $row[0] ** 2;
                $xty += $row[0] * $y[$i];
            }

            $xtx += $lambda;

            return $xtx > 1e-14 ? [$xty / $xtx] : [0.0];
        }

        try {
            $xMat = Matrix::quick($X);
            $yVec = Vector::quick($y);
            $xT   = $xMat->transpose();
            $xTx  = $xT->matmul($xMat);
            $xTy  = $xT->dot($yVec);

            if ($lambda > 0.0) {
                $reg = Matrix::diagonal(array_fill(0, $k, $lambda));
                $xTx = $xTx->add($reg);
            }

            return $xTx->inverse()->dot($xTy)->asArray();
        } catch (\Throwable $e) {
            return array_fill(0, $k, 0.0);
        }
    }

    /**
     * Solve a square linear system A x = b (used for Yule-Walker).
     *
     * @param float[][] $A
     * @param float[]   $b
     * @param float     $lambda Regularisation
     * @return float[]
     */
    protected function solveLinearSystem(array $A, array $b, float $lambda = 0.0) : array
    {
        $n = count($A);

        if ($n === 0) {
            return [];
        }

        if ($n === 1) {
            $a00 = $A[0][0] + $lambda;

            return $a00 > 1e-14 ? [$b[0] / $a00] : [0.0];
        }

        try {
            if ($lambda > 0.0) {
                for ($i = 0; $i < $n; $i++) {
                    $A[$i][$i] += $lambda;
                }
            }

            $aMat = Matrix::quick($A);
            $bVec = Vector::quick($b);

            return $aMat->inverse()->dot($bVec)->asArray();
        } catch (\Throwable $e) {
            return array_fill(0, $n, 0.0);
        }
    }

    /**
     * Replay the ARMA model on the centered training series and return the
     * trailing q residuals for seeding MA terms during forecasting.
     *
     * @param float[] $centered
     * @return float[]
     */
    protected function computeLastResiduals(array $centered) : array
    {
        if ($this->q === 0) {
            return [];
        }

        $n          = count($centered);
        $startIdx   = max($this->p, $this->q);
        $prevErrors = array_fill(0, $this->q, 0.0);

        for ($t = $startIdx; $t < $n; $t++) {
            $pred = 0.0;

            foreach ($this->phi as $i => $phi) {
                $pred += $phi * $centered[$t - 1 - $i];
            }

            foreach ($this->theta as $j => $theta) {
                $pred += $theta * $prevErrors[$j];
            }

            $error = $centered[$t] - $pred;

            array_unshift($prevErrors, $error);

            if (count($prevErrors) > $this->q) {
                array_pop($prevErrors);
            }
        }

        return $prevErrors;
    }

    /**
     * Return the string representation of this estimator.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'ARIMA (' . Params::stringify($this->params()) . ')';
    }
}
