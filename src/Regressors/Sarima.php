<?php

declare(strict_types=1);

namespace Rubix\ML\Regressors;

use Tensor\Matrix;
use Tensor\Vector;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Helpers\Params;
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
use function array_merge;

/**
 * SARIMA
 *
 * Seasonal AutoRegressive Integrated Moving Average (SARIMA) extends ARIMA
 * with multiplicative seasonal components. The full model notation is:
 *
 *   SARIMA(p, d, q)(P, D, Q)[s]
 *
 * where (P, D, Q) mirror the regular ARIMA orders but operate at seasonal
 * lags that are multiples of the seasonal period s.  The pipeline is:
 *
 *  1. Seasonal differencing  D times at lag s:  Δ_s^D y(t) = (1−B^s)^D y(t)
 *  2. Regular differencing   d times:           Δ^d w(t)
 *  3. Fit an extended ARMA whose AR lags include both 1…p and s, 2s, …, P·s,
 *     and whose MA lags include both 1…q and s, …, Q·s.
 *
 * **Training format**: same as ARIMA — `Labeled` dataset with labels as the
 * chronologically ordered time series.
 *
 * **Prediction format**: each sample must contain exactly `windowSize()`
 * original-scale observations in chronological order.
 *
 * **Forecasting**: call `forecast(int $steps)` for multi-step ahead prediction.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      The Rubix ML Community
 */
class Sarima extends Arima
{
    /**
     * Seasonal AR order.
     *
     * @var int<0,max>
     */
    protected int $bigP;

    /**
     * Seasonal differencing order.
     *
     * @var int<0,max>
     */
    protected int $bigD;

    /**
     * Seasonal MA order.
     *
     * @var int<0,max>
     */
    protected int $bigQ;

    /**
     * The seasonal period (e.g. 12 for monthly, 4 for quarterly, 7 for daily).
     *
     * @var positive-int
     */
    protected int $s;

    /**
     * Fitted seasonal AR coefficients Φ₁, …, ΦP (at lags s, 2s, …, P·s).
     *
     * @var float[]|null
     */
    protected ?array $bigPhi = null;

    /**
     * Fitted seasonal MA coefficients Θ₁, …, ΘQ (at lags s, …, Q·s).
     *
     * @var float[]|null
     */
    protected ?array $bigTheta = null;

    /**
     * @param int $p  Regular AR order
     * @param int $d  Regular differencing order
     * @param int $q  Regular MA order
     * @param int $P  Seasonal AR order
     * @param int $D  Seasonal differencing order
     * @param int $Q  Seasonal MA order
     * @param int $s  Seasonal period (must be ≥ 2)
     * @throws InvalidArgumentException
     */
    public function __construct(
        int $p = 1,
        int $d = 1,
        int $q = 0,
        int $P = 1,
        int $D = 1,
        int $Q = 0,
        int $s = 12
    ) {
        parent::__construct($p, $d, $q);

        if ($P < 0) {
            throw new InvalidArgumentException("Seasonal AR order P must be"
                . " non-negative, $P given.");
        }

        if ($D < 0) {
            throw new InvalidArgumentException("Seasonal differencing D must be"
                . " non-negative, $D given.");
        }

        if ($Q < 0) {
            throw new InvalidArgumentException("Seasonal MA order Q must be"
                . " non-negative, $Q given.");
        }

        if ($s < 2) {
            throw new InvalidArgumentException("Seasonal period s must be"
                . " at least 2, $s given.");
        }

        $this->bigP = $P;
        $this->bigD = $D;
        $this->bigQ = $Q;
        $this->s    = $s;
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
            'P' => $this->bigP,
            'D' => $this->bigD,
            'Q' => $this->bigQ,
            's' => $this->s,
        ];
    }

    /**
     * Has the model been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return parent::trained() && $this->bigPhi !== null;
    }

    /**
     * Return the number of features expected per sample in `predict()`.
     *
     * Window = seasonal context + regular context (after both differencings).
     *
     * @return int
     */
    public function windowSize() : int
    {
        // After D seasonal differences at lag s we need D*s extra original values,
        // then after d regular differences we need d more, then max(p, P*s) for AR.
        return $this->bigD * $this->s + $this->d + max($this->p, $this->bigP * $this->s);
    }

    /**
     * Fit the SARIMA model to a time series.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
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
        $n      = count($series);

        $longAR     = max(25, 2 * max($this->p, $this->q, $this->bigP, $this->bigQ) + 5);
        $minNeeded  = $this->bigD * $this->s + $this->d + $longAR
                    + max($this->p, $this->bigP * $this->s)
                    + max($this->q, $this->bigQ * $this->s) + 10;

        if ($n < $minNeeded) {
            throw new InvalidArgumentException(
                "Time series needs at least {$minNeeded} observations,"
                . " {$n} given."
            );
        }

        // Step 1: seasonal differencing (D times at lag s)
        $afterSeasonal = $this->applySeasonalDifference($series, $this->bigD, $this->s);

        // Step 2: regular differencing (d times)
        $differenced = $this->applyDifference($afterSeasonal, $this->d);

        // Step 3: center
        $this->mu = array_sum($differenced) / count($differenced);
        $centered = array_map(fn ($v) => $v - $this->mu, $differenced);

        // Step 4: fit extended ARMA
        $this->fitSarma($centered, $longAR);

        // Rolling buffer: last (D*s + d + max(p, P*s)) original values
        $bufLen       = $this->windowSize();
        $this->buffer = array_slice($series, -max($bufLen, 1));

        // Trailing residuals for MA seed
        $this->errors = $this->computeLastSarimaResiduals($centered);
    }

    /**
     * Predict the next value for each context window.
     *
     * @param Dataset $dataset
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
            $context       = array_map('floatval', $sample);
            $predictions[] = $this->sarimaPredictFromContext($context, []);
        }

        return $predictions;
    }

    /**
     * Generate future values from the end of the training series.
     *
     * @param int $steps
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
            $prediction  = $this->sarimaPredictFromContext($buffer, $errors);
            $forecasts[] = $prediction;

            $buffer[] = $prediction;
            $maxBuf   = $this->windowSize();

            while (count($buffer) > $maxBuf) {
                array_shift($buffer);
            }

            array_unshift($errors, 0.0);

            $maxErrors = max($this->q, $this->bigQ * $this->s, 1);

            if (count($errors) > $maxErrors) {
                array_pop($errors);
            }
        }

        return $forecasts;
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    /**
     * Predict from an original-scale context window with optional residuals.
     *
     * @param float[] $context
     * @param float[] $residuals
     * @return float
     */
    protected function sarimaPredictFromContext(array $context, array $residuals) : float
    {
        // Apply seasonal then regular differencing
        $afterSeasonal = $this->applySeasonalDifference(
            array_values($context),
            $this->bigD,
            $this->s
        );

        if ($this->d > 0) {
            $diffed = $this->applyDifference($afterSeasonal, $this->d);
        } else {
            $diffed = $afterSeasonal;
        }

        $centered = array_map(fn ($v) => $v - $this->mu, $diffed);
        $nC       = count($centered);

        // AR prediction (regular lags 1…p)
        $pred = 0.0;

        foreach ($this->phi as $i => $phi) {
            $idx = $nC - 1 - $i;

            if ($idx >= 0) {
                $pred += $phi * $centered[$idx];
            }
        }

        // Seasonal AR prediction (lags s, 2s, …, P·s)
        foreach ($this->bigPhi as $k => $bigPhi) {
            $lag = ($k + 1) * $this->s;
            $idx = $nC - $lag;

            if ($idx >= 0) {
                $pred += $bigPhi * $centered[$idx];
            }
        }

        // MA prediction (regular lags 1…q)
        foreach ($this->theta as $j => $theta) {
            $pred += $theta * ($residuals[$j] ?? 0.0);
        }

        // Seasonal MA prediction (lags s, …, Q·s)
        foreach ($this->bigTheta as $k => $bigTheta) {
            $lag = ($k + 1) * $this->s;
            $pred += $bigTheta * ($residuals[$lag - 1] ?? 0.0);
        }

        // Un-center
        $diffPred = $pred + $this->mu;

        // Invert regular differencing
        if ($this->d > 0) {
            $diffPred = $this->integrateDifference(
                $diffPred,
                $afterSeasonal,
                $this->d
            );
        }

        // Invert seasonal differencing
        if ($this->bigD > 0) {
            $diffPred = $this->integrateSeasonalDifference(
                $diffPred,
                array_values($context),
                $this->bigD,
                $this->s
            );
        }

        return $diffPred;
    }

    /**
     * Apply D-order seasonal differencing at lag s.
     *
     * Δ_s y(t) = y(t) − y(t−s)
     *
     * @param float[] $series
     * @param int     $D
     * @param int     $s
     * @return float[]
     */
    protected function applySeasonalDifference(array $series, int $D, int $s) : array
    {
        $result = $series;

        for ($iter = 0; $iter < $D; $iter++) {
            $n    = count($result);
            $next = [];

            for ($t = $s; $t < $n; $t++) {
                $next[] = $result[$t] - $result[$t - $s];
            }

            $result = $next;
        }

        return $result;
    }

    /**
     * Invert one step of D-th order seasonal differencing.
     *
     * Uses the same binomial inversion as regular differencing but
     * applied at lag s instead of lag 1.
     *
     * @param float   $w       Seasonally differenced prediction
     * @param float[] $context Original-scale context
     * @param int     $D
     * @param int     $s
     * @return float
     */
    protected function integrateSeasonalDifference(float $w, array $context, int $D, int $s) : float
    {
        $result = $w;
        $n      = count($context);

        for ($i = 1; $i <= $D; $i++) {
            $sign  = (($i % 2) === 1) ? 1.0 : -1.0;
            $binom = $this->binomialCoefficient($D, $i);
            $idx   = $n - $i * $s;

            if ($idx >= 0) {
                $result += $sign * $binom * $context[$idx];
            }
        }

        return $result;
    }

    /**
     * Fit the extended SARMA model.
     *
     * The design matrix includes both regular AR/MA lags (1…p, 1…q) and
     * seasonal AR/MA lags (s, 2s, …, P·s; s, …, Q·s).
     *
     * @param float[] $centered
     * @param int     $longAR
     */
    protected function fitSarma(array $centered, int $longAR) : void
    {
        $n     = count($centered);
        $maxAR = max($this->p, $this->bigP * $this->s, 1);
        $maxMA = max($this->q, $this->bigQ * $this->s, 1);

        // Stage 1: long AR for innovations
        $longARCoeffs = $this->fitLongAR($centered, min($longAR, (int) ($n / 3)));
        $m            = count($longARCoeffs);

        $innovations = array_fill(0, $n, 0.0);

        for ($t = $m; $t < $n; $t++) {
            $pred = 0.0;

            for ($i = 0; $i < $m; $i++) {
                $pred += $longARCoeffs[$i] * $centered[$t - 1 - $i];
            }

            $innovations[$t] = $centered[$t] - $pred;
        }

        // Collect all AR lags: regular 1…p, seasonal s, 2s, …, P·s
        $arLags = [];

        for ($i = 1; $i <= $this->p; $i++) {
            $arLags[] = $i;
        }

        for ($k = 1; $k <= $this->bigP; $k++) {
            $lag = $k * $this->s;

            if (!in_array($lag, $arLags)) {
                $arLags[] = $lag;
            }
        }

        // Collect all MA lags: regular 1…q, seasonal s, …, Q·s
        $maLags = [];

        for ($j = 1; $j <= $this->q; $j++) {
            $maLags[] = $j;
        }

        for ($k = 1; $k <= $this->bigQ; $k++) {
            $lag = $k * $this->s;

            if (!in_array($lag, $maLags)) {
                $maLags[] = $lag;
            }
        }

        $startIdx = max($maxAR, $maxMA, $m);
        $X        = [];
        $y        = [];

        if (empty($arLags) && empty($maLags)) {
            $this->phi      = [];
            $this->theta    = [];
            $this->bigPhi   = [];
            $this->bigTheta = [];

            return;
        }

        for ($t = $startIdx; $t < $n; $t++) {
            $row = [];

            foreach ($arLags as $lag) {
                $row[] = ($t - $lag >= 0) ? $centered[$t - $lag] : 0.0;
            }

            foreach ($maLags as $lag) {
                $row[] = ($t - $lag >= 0) ? $innovations[$t - $lag] : 0.0;
            }

            $X[] = $row;
            $y[] = $centered[$t];
        }

        if (empty($X)) {
            $nAR = count($arLags);
            $nMA = count($maLags);

            $this->phi      = array_fill(0, $this->p, 0.0);
            $this->theta    = array_fill(0, $this->q, 0.0);
            $this->bigPhi   = array_fill(0, $this->bigP, 0.0);
            $this->bigTheta = array_fill(0, $this->bigQ, 0.0);

            return;
        }

        $coeffs = $this->solveOLS($X, $y);

        // Assign coefficients back to regular and seasonal parts
        $offset = 0;

        $this->phi = [];

        for ($i = 0; $i < $this->p; $i++) {
            $this->phi[] = $coeffs[$offset++] ?? 0.0;
        }

        $this->bigPhi = [];

        for ($k = 0; $k < $this->bigP; $k++) {
            $seasonalLag = ($k + 1) * $this->s;

            // Skip if this lag was already in regular AR (avoid double-counting)
            if (!in_array($seasonalLag, array_slice($arLags, 0, $this->p))) {
                $this->bigPhi[] = $coeffs[$offset++] ?? 0.0;
            } else {
                $this->bigPhi[] = 0.0;
            }
        }

        $this->theta = [];

        for ($j = 0; $j < $this->q; $j++) {
            $this->theta[] = $coeffs[$offset++] ?? 0.0;
        }

        $this->bigTheta = [];

        for ($k = 0; $k < $this->bigQ; $k++) {
            $seasonalLag = ($k + 1) * $this->s;

            if (!in_array($seasonalLag, array_slice($maLags, 0, $this->q))) {
                $this->bigTheta[] = $coeffs[$offset++] ?? 0.0;
            } else {
                $this->bigTheta[] = 0.0;
            }
        }
    }

    /**
     * Compute trailing residuals for SARIMA MA seeding.
     *
     * @param float[] $centered
     * @return float[]
     */
    protected function computeLastSarimaResiduals(array $centered) : array
    {
        $maxMALag = max($this->q, $this->bigQ * $this->s, 1);

        if ($this->q === 0 && $this->bigQ === 0) {
            return [];
        }

        $n          = count($centered);
        $startIdx   = max($this->p, $this->bigP * $this->s, $this->q, $this->bigQ * $this->s, 1);
        $prevErrors = array_fill(0, $maxMALag, 0.0);

        for ($t = $startIdx; $t < $n; $t++) {
            $pred = 0.0;

            foreach ($this->phi as $i => $phi) {
                if ($t - 1 - $i >= 0) {
                    $pred += $phi * $centered[$t - 1 - $i];
                }
            }

            foreach ($this->bigPhi as $k => $bigPhi) {
                $lag = ($k + 1) * $this->s;

                if ($t - $lag >= 0) {
                    $pred += $bigPhi * $centered[$t - $lag];
                }
            }

            foreach ($this->theta as $j => $theta) {
                $pred += $theta * ($prevErrors[$j] ?? 0.0);
            }

            foreach ($this->bigTheta as $k => $bigTheta) {
                $lag = ($k + 1) * $this->s;
                $pred += $bigTheta * ($prevErrors[$lag - 1] ?? 0.0);
            }

            $error = $centered[$t] - $pred;

            array_unshift($prevErrors, $error);

            while (count($prevErrors) > $maxMALag) {
                array_pop($prevErrors);
            }
        }

        return $prevErrors;
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'SARIMA (' . Params::stringify($this->params()) . ')';
    }
}
