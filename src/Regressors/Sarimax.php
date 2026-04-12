<?php

declare(strict_types=1);

namespace Rubix\ML\Regressors;

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
use function array_map;
use function array_slice;
use function array_values;
use function array_sum;
use function array_fill;
use function array_merge;
use function in_array;
use function floatval;

/**
 * SARIMAX
 *
 * Seasonal AutoRegressive Integrated Moving Average with eXogenous variables.
 * Extends SARIMA(p,d,q)(P,D,Q)[s] by incorporating a contemporaneous linear
 * contribution from K exogenous regressors at each time step:
 *
 *   ŷ(t) = Σᵢ φᵢ y'(t−i)        [Regular AR, lags 1…p on differenced series]
 *         + Σₖ Φₖ y'(t−k·s)      [Seasonal AR, lags s, 2s, …, P·s]
 *         + Σⱼ θⱼ ε(t−j)         [Regular MA, lags 1…q]
 *         + Σₖ Θₖ ε(t−k·s)       [Seasonal MA, lags s, …, Q·s]
 *         + Σₗ βₗ x'ₗ(t)          [Exogenous regressors, differenced same as y]
 *         + μ                      [Mean of doubly-differenced series]
 *
 * where y'(t) = Δᵈ Δˢᴰ y(t) is the doubly-differenced endogenous series and
 * x'ₗ(t) = Δᵈ Δˢᴰ xₗ(t) applies the identical differencing pipeline to the
 * l-th exogenous variable so both live on the same stationary scale.
 *
 * =========================================================================
 * Coefficient estimation — Hannan–Rissanen two-stage OLS
 * =========================================================================
 *
 *   Stage 1 — Fit a high-order AR(m) model to the doubly-differenced,
 *              centred series y'(t) to obtain innovation approximations ε̂(t).
 *              AR order: m = max(25, 2·max(p,q,P,Q)+5), capped at n/3.
 *
 *   Stage 2 — Build the augmented design matrix and solve via ridge OLS:
 *
 *     X_row(t) = [ y'(t−l₁), …, y'(t−lₐᵣ),     ← combined AR lags (regular + seasonal)
 *                  ε̂(t−l₁), …, ε̂(t−lₘₐ),        ← stage-1 MA innovations (reg + seasonal)
 *                  x'₁(t),  …, x'ₖ(t)  ]          ← contemporaneous diff-X columns
 *
 *             β = (XᵀX + λI)⁻¹ Xᵀy
 *
 *   The coefficient vector is partitioned back into φ / Φ / θ / Θ / β blocks.
 *
 * =========================================================================
 * Training data contract
 * =========================================================================
 *
 *   `Labeled` dataset where:
 *     labels()  = chronologically ordered time-series values y(0), …, y(n−1)
 *     samples() = K raw (un-differenced) exogenous columns, one row per step:
 *                 sample[t] = [x₁(t), …, xₖ(t)]
 *
 *   If K = 0 any constant placeholder column is fine; SARIMAX degenerates to
 *   a pure SARIMA.
 *
 * =========================================================================
 * predict() data contract  ← CRITICAL: raw X, never pre-differenced
 * =========================================================================
 *
 *   Each sample must contain exactly predictWindowSize() = windowSize() + K
 *   values in this flat layout:
 *
 *     [ y(t−W), …, y(t−1),   x₁(t), …, xₖ(t) ]
 *      └──────────────────┘  └──────────────────┘
 *       windowSize() raw          K raw exogenous
 *       y-context window          at forecast horizon t
 *
 *   Both blocks are raw (original-scale).  The model differences them internally:
 *   y via the same pipeline as SARIMA, x via computeDiffXFromHistory() using the
 *   $xTail buffer retained from training.  Callers never pre-transform their data.
 *
 *   NOTE: because $xTail is a snapshot from the end of training, predict() is
 *   designed for one-step-ahead predictions that immediately follow the training
 *   boundary.  For multi-step horizons use forecast().
 *
 * =========================================================================
 * forecast() data contract
 * =========================================================================
 *
 *   forecast(int $steps, array $futureX = [])
 *
 *   $futureX[$step] = [x₁, …, xₖ]  — raw values for step t+step (0-indexed).
 *   Missing steps default to zero vectors.
 *   The model rolls a sliding X-history window forward, applying the same
 *   differencing pipeline at each step.
 *
 * =========================================================================
 * JIT optimisation contract
 * =========================================================================
 *
 *   1. All math arrays (innovations, diffXMatrix columns, design matrix rows,
 *      residual buffers) are pre-allocated with array_fill(0, $n, 0.0) before
 *      any loop writes into them — homogeneous float arrays for JIT compilation.
 *
 *   2. No array_shift / array_unshift inside hot loops.
 *      — forecast() buffer management uses array_slice (O(maxBuf) copy, no
 *        re-index scan) for the y-buffer, X-history, and MA residual buffers.
 *      — computeLastSarimaxResiduals() uses a pre-allocated ring buffer with
 *        O(1) modular-index writes/reads — eliminates the O(maxMALag) re-index
 *        per training step that array_unshift would impose.
 *
 *   3. unset() is called immediately after every large temporary array
 *      (design matrix X, y, innovations, diffXMatrix, per-column diff arrays,
 *      raw training samples, series) is no longer needed.
 *
 * References:
 * [1] G. E. P. Box, G. M. Jenkins, G. C. Reinsel, G. M. Ljung. (2015).
 *     Time Series Analysis: Forecasting and Control. 5th ed.
 * [2] J. D. Hamilton. (1994). Time Series Analysis. Princeton University Press.
 * [3] E. J. Hannan & B. G. Quinn. (1979). The determination of the order of
 *     an autoregression. Journal of the Royal Statistical Society B.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      The Rubix ML Community
 */
class Sarimax extends Sarima
{
    /**
     * Fitted exogenous regression coefficients β₁, …, βₖ.
     * Each coefficient scales the contemporaneous differenced exogenous column
     * contribution on the stationary (doubly-differenced) scale.
     *
     * @var float[]|null
     */
    protected ?array $betaX = null;

    /**
     * The number of exogenous features K discovered during training.
     * Stored so predict(), forecast(), and dimensionality checks remain valid
     * without inspecting $betaX directly.
     *
     * @var int<0,max>|null
     */
    protected ?int $numExogenous = null;

    /**
     * Sliding window of the last (D·s + d) raw exogenous rows from training.
     *
     * This buffer seeds the differencing operator in both predict() and
     * forecast().  Its length is exactly D·s + d (or 0 when no differencing
     * is required).  Appending one new raw-X row and applying Δᵈ Δˢᴰ yields
     * exactly one differenced-X vector — the contemporaneous x'(t).
     *
     * Stored as list<float[]> (one float[] of K values per row) so that
     * the JIT compiler sees a homogeneous nested-array structure.
     *
     * @var float[][]|null
     */
    protected ?array $xTail = null;

    /**
     * Return the hyper-parameter settings.
     * Adds the discovered exogenous feature count to the parent's parameter map.
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return array_merge(parent::params(), [
            'exogenous' => $this->numExogenous ?? 0,
        ]);
    }

    /**
     * Has the model been trained?
     * Requires parent SARIMA state AND fitted exogenous coefficients AND the
     * X-tail buffer.
     *
     * @return bool
     */
    public function trained() : bool
    {
        return parent::trained()
            && $this->betaX !== null
            && $this->xTail !== null;
    }

    /**
     * Return the total number of features expected per predict() sample.
     *
     *   predictWindowSize() = windowSize()    (raw y context, same as SARIMA)
     *                       + numExogenous    (raw X values at the forecast horizon)
     *
     * @return int
     */
    public function predictWindowSize() : int
    {
        return $this->windowSize() + ($this->numExogenous ?? 0);
    }

    /**
     * Fit the SARIMAX model to a labeled time-series dataset.
     *
     * Full pipeline:
     *
     *   1.  Extract time-series y from labels(); extract K exogenous columns
     *       from samples() — each row is one time step's raw X values.
     *   2.  Apply D-order seasonal differencing (lag s) to y → $afterSeasonal.
     *   3.  Apply d-order regular differencing to $afterSeasonal → $differenced.
     *   4.  Centre the doubly-differenced series; store the mean μ.
     *   5.  Identical differencing pipeline applied to each X column to produce
     *       $diffXMatrix[K][n−D·s−d] — same time alignment as $centered.
     *   6.  Fit the extended SARMAX via fitSarimax() (Hannan–Rissanen two-stage
     *       OLS augmented with K contemporaneous diff-X columns).
     *   7.  Store y forecast buffer $this->buffer (same as Sarima::train()).
     *   8.  Store X-tail buffer $this->xTail: the last (D·s + d) raw X rows.
     *       This seeds computeDiffXFromHistory() during predict() and forecast().
     *   9.  Replay the fitted model on the centred training series to compute
     *       trailing residuals for MA-term seeding ($this->errors).
     *
     * @param Dataset $dataset
     * @throws InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        SpecificationChain::with([
            new DatasetIsLabeled($dataset),
            new DatasetIsNotEmpty($dataset),
        ])->check();

        /** @var Labeled $dataset */
        $rawSamples = $dataset->samples();
        $series     = array_values(array_map('floatval', $dataset->labels()));
        $n          = count($series);

        // Detect K from the first sample row.
        // K = 0 → SARIMAX degenerates to a pure SARIMA.
        $numX               = !empty($rawSamples[0]) ? count($rawSamples[0]) : 0;
        $this->numExogenous = $numX;

        // ------------------------------------------------------------------
        // Minimum-observations guard (mirrors Sarima::train()).
        // ------------------------------------------------------------------
        $longAR    = max(25, 2 * max($this->p, $this->q, $this->bigP, $this->bigQ) + 5);
        $minNeeded = $this->bigD * $this->s
            + $this->d
            + $longAR
            + max($this->p, $this->bigP * $this->s)
            + max($this->q, $this->bigQ * $this->s)
            + 10;

        if ($n < $minNeeded) {
            throw new InvalidArgumentException(
                "Time series needs at least {$minNeeded} observations, {$n} given."
            );
        }

        // ------------------------------------------------------------------
        // Step 1–3: Double-difference y and centre.
        //
        // applySeasonalDifference() consumes D·s leading values.
        // applyDifference() consumes d more, giving count($centered) = n - D·s - d.
        // ------------------------------------------------------------------
        $afterSeasonal = $this->applySeasonalDifference($series, $this->bigD, $this->s);
        $differenced   = $this->applyDifference($afterSeasonal, $this->d);

        unset($afterSeasonal);

        $this->mu = array_sum($differenced) / count($differenced);
        $centered = array_map(fn ($v) => $v - $this->mu, $differenced);

        unset($differenced);

        // ------------------------------------------------------------------
        // Step 4: Extract and double-difference each exogenous column.
        //
        // Each raw X column is extracted into a pre-allocated homogeneous float
        // array ($xCol) before the differencing calls, so the JIT compiler has
        // a fixed-size known-type target.  After differencing, each column in
        // $diffXMatrix has the same length as $centered (count = n - D·s - d),
        // preserving time alignment for the Stage-2 design matrix.
        // ------------------------------------------------------------------

        /** @var list<float[]> $diffXMatrix  Differenced X columns, shape [K][n−D·s−d]. */
        $diffXMatrix = [];

        if ($numX > 0) {
            for ($col = 0; $col < $numX; ++$col) {
                /** @var float[] $xCol  Pre-allocated homogeneous float column. */
                $xCol = array_fill(0, $n, 0.0);

                for ($i = 0; $i < $n; ++$i) {
                    $xCol[$i] = (float) ($rawSamples[$i][$col] ?? 0.0);
                }

                $xSeasoned = $this->applySeasonalDifference($xCol, $this->bigD, $this->s);

                unset($xCol);

                $xDiff = $this->applyDifference($xSeasoned, $this->d);

                unset($xSeasoned);

                $diffXMatrix[$col] = $xDiff;

                unset($xDiff);
            }
        }

        // ------------------------------------------------------------------
        // Step 5: Fit the extended SARMAX model.
        // ------------------------------------------------------------------
        $this->fitSarimax($centered, $longAR, $diffXMatrix);

        // ------------------------------------------------------------------
        // Step 6: Store y forecast buffer (identical to Sarima::train()).
        // ------------------------------------------------------------------
        $bufLen       = $this->windowSize();
        $this->buffer = array_slice($series, -max($bufLen, 1));

        // ------------------------------------------------------------------
        // Step 7: Store X-tail buffer.
        //
        // $xTail holds the last (D·s + d) raw X rows from the training set.
        // During predict() or forecast(), a new raw X row is appended to this
        // tail to form a window of (D·s + d + 1) rows; applying Δᵈ Δˢᴰ to
        // each column then yields exactly one differenced-X value.
        //
        // Proof: (D·s + d + 1) rows
        //        − D seasonal diffs (each removes s rows): D·s removed → d + 1 remain
        //        − d regular diffs (removes d rows):        d removed   → 1 remains  ✓
        //
        // Layout: $xTail[0] = oldest retained row, $xTail[xTailLen−1] = newest.
        // Each row is a pre-allocated float[K] to maintain homogeneous structure.
        // ------------------------------------------------------------------
        $xTailLen = $this->bigD * $this->s + $this->d;

        if ($numX > 0 && $xTailLen > 0) {
            $rawTailRows = array_slice($rawSamples, -$xTailLen);
            $this->xTail = [];

            foreach ($rawTailRows as $row) {
                /** @var float[] $floatRow */
                $floatRow = array_fill(0, $numX, 0.0);

                for ($col = 0; $col < $numX; ++$col) {
                    $floatRow[$col] = (float) ($row[$col] ?? 0.0);
                }

                $this->xTail[] = $floatRow;
            }

            unset($rawTailRows, $floatRow);
        } else {
            // No differencing or K = 0: no tail history needed.
            $this->xTail = [];
        }

        // ------------------------------------------------------------------
        // Step 8: Replay the model on the training series to compute the
        //         trailing residuals that seed MA terms during forecasting.
        // ------------------------------------------------------------------
        $this->errors = $this->computeLastSarimaxResiduals($centered, $diffXMatrix);

        // Release large working arrays — training is complete.
        unset($centered, $diffXMatrix, $rawSamples, $series);
    }

    /**
     * Predict the next value for each context window in the dataset.
     *
     * Each sample must contain predictWindowSize() = windowSize() + numExogenous
     * values in this flat layout:
     *
     *   [ y(t−W), …, y(t−1),   x₁(t), …, xₖ(t) ]
     *
     * Both blocks are raw (un-transformed, original-scale).  The model
     * differences both internally.  Callers never pre-transform their data.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<float>
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->trained()) {
            throw new \RuntimeException('Estimator has not been trained.');
        }

        $numX = $this->numExogenous ?? 0;
        $ws   = $this->windowSize();

        // 1. Dimensionality Check: We ONLY expect the exogenous variables from the caller.
        if ($numX > 0) {
            DatasetHasDimensionality::with($dataset, $numX)->check();
        }

        $predictions = [];

        // 2. Initialize rolling buffers from the state saved during train()
        // This eliminates the need for the user to manually pass historical context.
        $context  = $this->yTail ?? []; 
        $xHistory = $this->xTail ?? [];

        foreach ($dataset->samples() as $sample) {
            
            // 3. Extract ONLY the exogenous variables from the sample
            /** @var float[] $rawX */
            $rawX = array_fill(0, $numX, 0.0);
            for ($col = 0; $col < $numX; ++$col) {
                $rawX[$col] = (float) ($sample[$col] ?? 0.0);
            }

            // 4. Difference X and calculate the prediction
            $diffX = $this->computeDiffXFromHistory($rawX, $xHistory);
            
            // Generate the forecast for this time step
            $yHat  = $this->sarimaxPredictFromContext($context, [], $diffX);
            $predictions[] = $yHat;

            // ------------------------------------------------------------------
            // Stateful Rolling: Move the windows forward by 1 time step
            // ------------------------------------------------------------------
            
            // Roll the y context (Drop oldest, append the new prediction)
            if ($ws > 0) {
                $context = array_slice($context, 1);
                $context[] = $yHat;
            }

            // Roll the X history (needed for differencing the next step)
            if (count($xHistory) > 0) {
                $xHistory = array_slice($xHistory, 1);
                $xHistory[] = $rawX;
            }
        }

        return $predictions;
    }

    /**
     * Generate future values from the end of the training series.
     *
     * Rolling-buffer algorithm:
     *   1. At each step, resolve raw X (from $futureX or zeros).
     *   2. Difference X via computeDiffXFromHistory($rawX, $xHistory).
     *   3. Produce one-step prediction via sarimaxPredictFromContext().
     *   4. Advance y-buffer, X-history, and MA residual buffer forward.
     *
     * Buffer management uses array_slice instead of array_shift to avoid
     * O(n) re-indexing.  For the MA residual buffer, a pre-allocated shift
     * loop is used to prepend 0.0 (unknown future shock) without array_unshift.
     *
     * @param int       $steps   Number of steps ahead to forecast (≥ 1).
     * @param float[][] $futureX Raw (un-differenced) exogenous values for each
     *                           forecast step: $futureX[$step][0..K−1].
     *                           Missing steps default to zero vectors.
     * @throws RuntimeException
     * @throws InvalidArgumentException
     * @return float[]
     */
    public function forecast(int $steps = 1, array $futureX = []) : array
    {
        if (!$this->trained()) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        if ($steps < 1) {
            throw new InvalidArgumentException(
                "Steps must be at least 1, {$steps} given."
            );
        }

        $numX      = $this->numExogenous ?? 0;
        $forecasts = [];
        $buffer    = $this->buffer;         // rolling y-context window (original scale)
        $errors    = $this->errors ?? [];   // rolling MA residual seed
        $xHistory  = $this->xTail ?? [];    // rolling raw X-history window

        $maxBuf    = $this->windowSize();
        $maxXHist  = max($this->bigD * $this->s + $this->d, 1);
        $maxErrors = max($this->q, $this->bigQ * $this->s, 1);

        for ($step = 0; $step < $steps; ++$step) {

            // ------------------------------------------------------------------
            // Resolve raw X for this step (zeros if the caller omitted it).
            // Pre-allocate as homogeneous float array for JIT stability.
            // ------------------------------------------------------------------

            /** @var float[] $rawX */
            $rawX = array_fill(0, $numX, 0.0);

            if (isset($futureX[$step])) {
                for ($col = 0; $col < $numX; ++$col) {
                    $rawX[$col] = (float) ($futureX[$step][$col] ?? 0.0);
                }
            }

            // Compute differenced X from the rolling X-history window.
            $diffX = $this->computeDiffXFromHistory($rawX, $xHistory);

            // One-step SARIMAX prediction.
            $prediction  = $this->sarimaxPredictFromContext($buffer, $errors, $diffX);
            $forecasts[] = $prediction;

            unset($diffX);

            // ------------------------------------------------------------------
            // Advance the y-buffer: append new prediction, drop oldest if needed.
            // array_slice is O(maxBuf) and avoids the O(n) re-index of array_shift.
            // ------------------------------------------------------------------
            $buffer[] = $prediction;

            if (count($buffer) > $maxBuf) {
                $buffer = array_slice($buffer, -$maxBuf);
            }

            // ------------------------------------------------------------------
            // Advance X-history: append new raw X row, trim to maxXHist.
            // ------------------------------------------------------------------
            if ($numX > 0) {
                $xHistory[] = $rawX;

                if (count($xHistory) > $maxXHist) {
                    $xHistory = array_slice($xHistory, -$maxXHist);
                }
            }

            unset($rawX);

            // ------------------------------------------------------------------
            // Advance the MA residual buffer: prepend 0.0 (unknown future shock;
            // minimum-MSE point forecast) and trim to maxErrors length.
            //
            // Implemented as a pre-allocated copy loop to avoid array_unshift's
            // O(maxErrors) re-index overhead per step.
            // ------------------------------------------------------------------

            /** @var float[] $newErrors  Pre-allocated homogeneous float buffer. */
            $newErrors = array_fill(0, $maxErrors, 0.0);

            // $newErrors[0] = 0.0 (newest, unknown shock already set by array_fill).
            // Copy old errors into positions [1..maxErrors-1].
            $copyLen = min(count($errors), $maxErrors - 1);

            for ($j = 0; $j < $copyLen; ++$j) {
                $newErrors[$j + 1] = $errors[$j];
            }

            $errors = $newErrors;

            unset($newErrors);
        }

        return $forecasts;
    }

    // =========================================================================
    // Internal helpers
    // =========================================================================

    /**
     * Fit the extended SARMAX model via Hannan–Rissanen two-stage OLS.
     *
     * Stage 1 — long AR(m) model on the centred series to approximate innovations.
     *
     * Stage 2 — build the augmented design matrix and solve via ridge OLS:
     *
     *   X_row(t) = [ y'(t−l₁), …, y'(t−lₐᵣ),    ← combined AR lags (reg + seasonal)
     *                ε̂(t−l₁), …, ε̂(t−lₘₐ),       ← MA innovations  (reg + seasonal)
     *                x'₁(t),  …, x'ₖ(t) ]          ← contemporaneous diff-X
     *
     * The solved β vector is partitioned back into
     *   φ (regular AR) / Φ (seasonal AR) / θ (regular MA) / Θ (seasonal MA) / β (exogenous).
     *
     * JIT notes:
     *   - $innovations pre-allocated with array_fill, overwritten in place.
     *   - Design matrix $X and RHS $y are unset immediately after solveOLS().
     *   - $betaX pre-allocated as a homogeneous float array before extraction.
     *   - $longARCoeffs unset as soon as innovations are computed.
     *
     * @param float[]   $centered    Centred doubly-differenced time series.
     * @param int       $longAR      Stage-1 long-AR order.
     * @param float[][] $diffXMatrix Differenced X columns, shape [K][n−D·s−d].
     */
    protected function fitSarimax(array $centered, int $longAR, array $diffXMatrix) : void
    {
        $n    = count($centered);
        $numX = count($diffXMatrix);

        $maxAR = max($this->p, $this->bigP * $this->s, 1);
        $maxMA = max($this->q, $this->bigQ * $this->s, 1);

        // ------------------------------------------------------------------
        // Stage 1: Long AR approximation for innovation estimation.
        //
        // fitLongAR() solves the Yule-Walker equations for AR(m) coefficients.
        // We then replay the long AR in a single forward pass to fill $innovations.
        // ------------------------------------------------------------------
        $longARCoeffs = $this->fitLongAR($centered, min($longAR, (int) ($n / 3)));
        $m            = count($longARCoeffs);

        /** @var float[] $innovations  Pre-allocated homogeneous float array. */
        $innovations = array_fill(0, $n, 0.0);

        for ($t = $m; $t < $n; ++$t) {
            $pred = 0.0;

            for ($i = 0; $i < $m; ++$i) {
                $pred += $longARCoeffs[$i] * $centered[$t - 1 - $i];
            }

            $innovations[$t] = $centered[$t] - $pred;
        }

        // Long-AR coefficients no longer needed after innovations are computed.
        unset($longARCoeffs);

        // ------------------------------------------------------------------
        // Collect combined AR lag set: regular 1…p + seasonal s, 2s, …, P·s
        // (deduplicated using in_array — same logic as Sarima::fitSarma()).
        // ------------------------------------------------------------------
        $arLags = [];

        for ($i = 1; $i <= $this->p; ++$i) {
            $arLags[] = $i;
        }

        for ($k = 1; $k <= $this->bigP; ++$k) {
            $lag = $k * $this->s;

            if (!in_array($lag, $arLags)) {
                $arLags[] = $lag;
            }
        }

        // ------------------------------------------------------------------
        // Collect combined MA lag set: regular 1…q + seasonal s, …, Q·s.
        // ------------------------------------------------------------------
        $maLags = [];

        for ($j = 1; $j <= $this->q; ++$j) {
            $maLags[] = $j;
        }

        for ($k = 1; $k <= $this->bigQ; ++$k) {
            $lag = $k * $this->s;

            if (!in_array($lag, $maLags)) {
                $maLags[] = $lag;
            }
        }

        // Trivial case: no AR, MA, or X terms → zero-fill all coefficient arrays.
        if (empty($arLags) && empty($maLags) && $numX === 0) {
            $this->phi      = [];
            $this->theta    = [];
            $this->bigPhi   = [];
            $this->bigTheta = [];
            $this->betaX    = [];

            return;
        }

        // ------------------------------------------------------------------
        // Stage 2: Build the augmented OLS design matrix.
        //
        // startIdx ensures every AR/MA lag reference is in-bounds.
        // The K exogenous columns are appended contemporaneously (same index t).
        // ------------------------------------------------------------------
        $startIdx = max($maxAR, $maxMA, $m);

        $X = [];
        $y = [];

        for ($t = $startIdx; $t < $n; ++$t) {
            $row = [];

            // Combined AR lags: y'(t−lag) for each lag in the dedup'd set.
            foreach ($arLags as $lag) {
                $row[] = ($t - $lag >= 0) ? $centered[$t - $lag] : 0.0;
            }

            // Combined MA lags: ε̂(t−lag) for each lag in the dedup'd set.
            foreach ($maLags as $lag) {
                $row[] = ($t - $lag >= 0) ? $innovations[$t - $lag] : 0.0;
            }

            // Contemporaneous differenced exogenous columns.
            for ($col = 0; $col < $numX; ++$col) {
                $row[] = (float) ($diffXMatrix[$col][$t] ?? 0.0);
            }

            $X[] = $row;
            $y[] = $centered[$t];
        }

        // Innovations no longer needed — free before the large OLS solve.
        unset($innovations);

        // Empty design matrix guard (series too short for startIdx).
        if (empty($X)) {
            $this->phi      = array_fill(0, $this->p, 0.0);
            $this->bigPhi   = array_fill(0, $this->bigP, 0.0);
            $this->theta    = array_fill(0, $this->q, 0.0);
            $this->bigTheta = array_fill(0, $this->bigQ, 0.0);
            $this->betaX    = array_fill(0, $numX, 0.0);

            return;
        }

        // Solve the ridge-regularised OLS normal equations β = (XᵀX + λI)⁻¹ Xᵀy.
        $coeffs = $this->solveOLS($X, $y);

        // Design matrix and RHS freed immediately after solve — can be large.
        unset($X, $y);

        // ------------------------------------------------------------------
        // Partition the flat coefficient vector back into named blocks.
        //
        // Column order in the design matrix (and coefficient vector):
        //   [regular AR (p cols)]  [seasonal AR (bigP cols, skipping overlaps)]
        //   [regular MA (q cols)]  [seasonal MA (bigQ cols, skipping overlaps)]
        //   [exogenous  (numX cols)]
        //
        // "Skipping overlaps": if a seasonal lag k·s already appeared in the
        // regular AR/MA block (i.e., k·s ≤ p or k·s ≤ q), no column was added
        // to the design matrix for it, so we store 0.0 without consuming an offset.
        // ------------------------------------------------------------------
        $offset = 0;

        // Regular AR: φ₁ … φₚ
        $this->phi = [];

        for ($i = 0; $i < $this->p; ++$i) {
            $this->phi[] = (float) ($coeffs[$offset++] ?? 0.0);
        }

        // Seasonal AR: Φ₁ … ΦP — skip if the lag was already in regular AR.
        $regularArLags = array_slice($arLags, 0, $this->p);

        $this->bigPhi = [];

        for ($k = 0; $k < $this->bigP; ++$k) {
            $seasonalLag = ($k + 1) * $this->s;

            $this->bigPhi[] = !in_array($seasonalLag, $regularArLags)
                ? (float) ($coeffs[$offset++] ?? 0.0)
                : 0.0;
        }

        // Regular MA: θ₁ … θq
        $this->theta = [];

        for ($j = 0; $j < $this->q; ++$j) {
            $this->theta[] = (float) ($coeffs[$offset++] ?? 0.0);
        }

        // Seasonal MA: Θ₁ … ΘQ — same overlap-skip logic as seasonal AR.
        $regularMaLags = array_slice($maLags, 0, $this->q);

        $this->bigTheta = [];

        for ($k = 0; $k < $this->bigQ; ++$k) {
            $seasonalLag = ($k + 1) * $this->s;

            $this->bigTheta[] = !in_array($seasonalLag, $regularMaLags)
                ? (float) ($coeffs[$offset++] ?? 0.0)
                : 0.0;
        }

        // Exogenous: β₁ … βₖ — pre-allocate as homogeneous float array.
        /** @var float[] $betaX */
        $betaX = array_fill(0, $numX, 0.0);

        for ($col = 0; $col < $numX; ++$col) {
            $betaX[$col] = (float) ($coeffs[$offset++] ?? 0.0);
        }

        $this->betaX = $betaX;

        unset($coeffs, $betaX);
    }

    /**
     * Produce a one-step SARIMAX prediction from a raw y context window,
     * optional MA residuals, and **already-differenced** exogenous values.
     *
     * Called internally only (by predict() and forecast()) after the
     * raw-X differencing step has been performed via computeDiffXFromHistory().
     * Mirrors Sarima::sarimaPredictFromContext() exactly, with the exogenous
     * term Σₗ βₗ x'ₗ(t) added on the differenced/centred scale before the
     * prediction is un-centred and the integration is inverted.
     *
     * @param float[] $context   Raw y history of length windowSize().
     * @param float[] $residuals Trailing MA residuals, [0]=newest (empty → 0).
     * @param float[] $diffX     K differenced exogenous values at horizon t.
     * @return float
     */
    protected function sarimaxPredictFromContext(
        array $context,
        array $residuals,
        array $diffX
    ) : float {
        // Apply seasonal then regular differencing to the raw y context window.
        $afterSeasonal = $this->applySeasonalDifference(
            array_values($context),
            $this->bigD,
            $this->s
        );

        $diffed = ($this->d > 0)
            ? $this->applyDifference($afterSeasonal, $this->d)
            : $afterSeasonal;

        $centered = array_map(fn ($v) => $v - $this->mu, $diffed);
        $nC       = count($centered);

        unset($diffed);

        // Regular AR: Σᵢ φᵢ · y'(t−i).
        $pred = 0.0;

        foreach ($this->phi as $i => $phi) {
            $idx = $nC - 1 - $i;

            if ($idx >= 0) {
                $pred += $phi * $centered[$idx];
            }
        }

        // Seasonal AR: Σₖ Φₖ · y'(t−k·s).
        foreach ($this->bigPhi as $k => $bigPhi) {
            $lag = ($k + 1) * $this->s;
            $idx = $nC - $lag;

            if ($idx >= 0) {
                $pred += $bigPhi * $centered[$idx];
            }
        }

        unset($centered);

        // Regular MA: Σⱼ θⱼ · ε(t−j).
        // $residuals[0] = most recent error (lag 1), $residuals[j] = lag j+1.
        foreach ($this->theta as $j => $theta) {
            $pred += $theta * ($residuals[$j] ?? 0.0);
        }

        // Seasonal MA: Σₖ Θₖ · ε(t−k·s).
        // lag = (k+1)·s, 1-indexed access: $residuals[$lag - 1].
        foreach ($this->bigTheta as $k => $bigTheta) {
            $lag = ($k + 1) * $this->s;
            $pred += $bigTheta * ($residuals[$lag - 1] ?? 0.0);
        }

        // Exogenous: Σₗ βₗ · x'ₗ(t).
        // Added on the differenced scale, consistent with all AR/MA terms.
        foreach ($this->betaX as $col => $beta) {
            $pred += $beta * ($diffX[$col] ?? 0.0);
        }

        // Un-centre, then invert regular differencing.
        $diffPred = $pred + $this->mu;

        if ($this->d > 0) {
            $diffPred = $this->integrateDifference($diffPred, $afterSeasonal, $this->d);
        }

        unset($afterSeasonal);

        // Invert seasonal differencing.
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
     * Replay the fitted model on the centred training series to obtain the
     * trailing residuals that seed MA terms at the start of forecasting.
     *
     * Mirrors Sarima::computeLastSarimaResiduals() but includes the exogenous
     * term in the one-step prediction replayed at each time step t:
     *
     *   ε(t) = y'(t) − ΣAR − ΣMA − Σₗ βₗ x'ₗ(t)
     *
     * =========================================================================
     * Ring-buffer implementation — O(1) per step
     * =========================================================================
     *
     * The naive approach (array_unshift each step) is O(maxMALag) per step and
     * destroys JIT scalar-array optimisations.  Instead we use a pre-allocated
     * circular buffer of length maxMALag:
     *
     *   $errorRing[0..maxMALag−1]  — fixed-size pre-allocated float array.
     *   $errorHead                  — index where the NEXT error will be written;
     *                                 after writing: advance head = (head+1)%max.
     *
     * Reading lag-j (1-indexed) after writing up to t−1:
     *   errorRing[($errorHead + $maxMALag − $j) % $maxMALag]
     *
     * Reading lag as used by theta[$j] (0-indexed, corresponds to lag j+1):
     *   errorRing[($errorHead + $maxMALag − 1 − $j) % $maxMALag]
     *
     * At the end of the loop, $prevErrors[0..maxMALag−1] is reconstructed in
     * the linear format ([0]=newest) expected by sarimaxPredictFromContext().
     *
     * @param float[]   $centered    Centred doubly-differenced training series.
     * @param float[][] $diffXMatrix Differenced X columns, shape [K][n−D·s−d].
     * @return float[]               Trailing residuals, [0]=newest, length=maxMALag.
     */
    protected function computeLastSarimaxResiduals(
        array $centered,
        array $diffXMatrix
    ) : array {
        $numX     = count($diffXMatrix);
        $maxMALag = max($this->q, $this->bigQ * $this->s, 1);

        // Pure AR model: no residuals needed.
        if ($this->q === 0 && $this->bigQ === 0) {
            return [];
        }

        $n        = count($centered);
        $startIdx = max(
            $this->p,
            $this->bigP * $this->s,
            $this->q,
            $this->bigQ * $this->s,
            1
        );

        // ------------------------------------------------------------------
        // Pre-allocated ring buffer.
        //
        // $errorHead = next write position (initialised to 0).
        // All slots start at 0.0, which provides the correct zero-initialisation
        // for MA terms that reference steps before startIdx.
        // ------------------------------------------------------------------

        /** @var float[] $errorRing */
        $errorRing = array_fill(0, $maxMALag, 0.0);
        $errorHead = 0;

        for ($t = $startIdx; $t < $n; ++$t) {
            $pred = 0.0;

            // Regular AR.
            foreach ($this->phi as $i => $phi) {
                if ($t - 1 - $i >= 0) {
                    $pred += $phi * $centered[$t - 1 - $i];
                }
            }

            // Seasonal AR.
            foreach ($this->bigPhi as $k => $bigPhi) {
                $lag = ($k + 1) * $this->s;

                if ($t - $lag >= 0) {
                    $pred += $bigPhi * $centered[$t - $lag];
                }
            }

            // Regular MA: theta[j] (0-indexed) = coeff for lag j+1.
            // Ring read: error (j+1) steps ago = ring[(head + max − 1 − j) % max].
            foreach ($this->theta as $j => $theta) {
                $pred += $theta * $errorRing[($errorHead + $maxMALag - 1 - $j) % $maxMALag];
            }

            // Seasonal MA: bigTheta[k] = coeff for lag (k+1)·s.
            // Ring read: error $lag steps ago = ring[(head + max − lag) % max].
            foreach ($this->bigTheta as $k => $bigTheta) {
                $lag = ($k + 1) * $this->s;
                $pred += $bigTheta * $errorRing[($errorHead + $maxMALag - $lag) % $maxMALag];
            }

            // Exogenous contribution on the differenced scale.
            for ($col = 0; $col < $numX; ++$col) {
                $pred += ($this->betaX[$col] ?? 0.0) * ($diffXMatrix[$col][$t] ?? 0.0);
            }

            // Write new residual to the ring at the current head, then advance.
            // This overwrites the oldest slot (the one that is now out of window).
            $errorRing[$errorHead] = $centered[$t] - $pred;
            $errorHead = ($errorHead + 1) % $maxMALag;
        }

        // ------------------------------------------------------------------
        // Convert ring buffer → linear format.
        //
        // After the loop $errorHead points to the next write position, meaning
        // the newest residual sits at (errorHead − 1 + maxMALag) % maxMALag.
        //
        // $prevErrors[j] = residual from j+1 steps ago (lag j+1):
        //   ring[($errorHead + $maxMALag − 1 − j) % $maxMALag]
        // ------------------------------------------------------------------

        /** @var float[] $prevErrors */
        $prevErrors = array_fill(0, $maxMALag, 0.0);

        for ($j = 0; $j < $maxMALag; ++$j) {
            $prevErrors[$j] = $errorRing[($errorHead + $maxMALag - 1 - $j) % $maxMALag];
        }

        unset($errorRing);

        return $prevErrors;
    }

    /**
     * Compute the differenced X vector for a single prediction step.
     *
     * Prepends $xHistory (the last D·s + d raw rows from recent history) to a
     * single new raw X row, then applies the same Δᵈ Δˢᴰ pipeline used during
     * training.  The resulting sequence has exactly one element — the
     * contemporaneous x'(t) value needed by sarimaxPredictFromContext().
     *
     * Proof of output length:
     *   fullHistory = D·s + d + 1 rows
     *   after D seasonal diffs (each removes s rows): D·s removed → d + 1 remain
     *   after d regular diffs (removes d more):        d removed   → 1 remains  ✓
     *
     * Short-circuits when d = 0 and D = 0: returns $rawX as-is without any
     * allocation or loop — the JIT compiler will specialise this common case.
     *
     * Memory note: three small float arrays ($colVals, $xSeasoned, $xDiff) are
     * allocated and immediately unset per-column per-call, keeping the live heap
     * footprint bounded by K × (D·s + d + 1) floats.
     *
     * @param float[]   $rawX      K raw exogenous values at the prediction horizon.
     * @param float[][] $xHistory  Last (D·s + d) raw X rows from recent history.
     * @return float[]             K differenced exogenous values.
     */
    protected function computeDiffXFromHistory(array $rawX, array $xHistory) : array
    {
        $numX = count($rawX);

        /** @var float[] $diffX  Pre-allocated homogeneous output. */
        $diffX = array_fill(0, $numX, 0.0);

        // Fast path: no differencing required (d = 0 and D = 0).
        if ($this->d === 0 && $this->bigD === 0) {
            for ($col = 0; $col < $numX; ++$col) {
                $diffX[$col] = (float) $rawX[$col];
            }

            return $diffX;
        }

        // Build the combined history-plus-current window.
        // Shape: (D·s + d + 1) × K — after differencing: 1 × K.
        $fullHistory = array_merge($xHistory, [$rawX]);
        $histLen     = count($fullHistory);

        for ($col = 0; $col < $numX; ++$col) {
            // Extract this column as a pre-allocated homogeneous float array.
            /** @var float[] $colVals */
            $colVals = array_fill(0, $histLen, 0.0);

            for ($row = 0; $row < $histLen; ++$row) {
                $colVals[$row] = (float) ($fullHistory[$row][$col] ?? 0.0);
            }

            // Seasonal differencing: consumes D·s leading rows.
            $xSeasoned = $this->applySeasonalDifference($colVals, $this->bigD, $this->s);

            unset($colVals);

            // Regular differencing: consumes d more rows → 1 remains.
            $xDiff = $this->applyDifference($xSeasoned, $this->d);

            unset($xSeasoned);

            // The sole remaining element is the differenced x'ₗ(t).
            $diffX[$col] = (float) ($xDiff[count($xDiff) - 1] ?? 0.0);

            unset($xDiff);
        }

        return $diffX;
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'SARIMAX (' . Params::stringify($this->params()) . ')';
    }
}
