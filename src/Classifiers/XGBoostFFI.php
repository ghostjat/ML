<?php

declare(strict_types=1);

namespace Rubix\ML\Classifiers;

use FFI;
use InvalidArgumentException;
use RuntimeException;
use Rubix\ML\Blas\BLASFFI;
use Rubix\ML\Datasets\ArrowDataset;

/**
 * XGBoostFFI
 *
 * XGBoost gradient-boosted-trees classifier exposed as a PHP class via
 * direct PHP-FFI bindings to the official XGBoost C API (xgboost/c_api.h).
 *
 * Zero-copy data path
 * ───────────────────
 * Features:
 *   ArrowDataset::getFeaturePointer() returns a BLAS-scope double[rows * cols].
 *   XGDMatrixCreateFromMat() requires const float*, so a single O(rows × cols)
 *   cast pass copies double → float into a temporary buffer allocated in the
 *   XGBoost FFI scope (ensuring type compatibility).  The buffer is freed
 *   immediately after XGBoost creates the DMatrix with its own internal copy.
 *
 * Labels:
 *   Similarly cast from ArrowDataset's float[rows] (BLAS scope) into a fresh
 *   XGBoost-scope float[rows] temp buffer, then passed to XGDMatrixSetFloatInfo.
 *   The temp buffer is freed immediately after XGBoost ingests the values.
 *
 * Memory lifecycle
 * ────────────────
 *   train()    → DMatrix handle + Booster handle stored as instance state.
 *   predict()  → Creates a local predict-DMatrix, freed after XGBoosterPredict.
 *   __destruct → XGBoosterFree(booster) then XGDMatrixFree(trainDMatrix).
 *
 * Handle storage
 * ──────────────
 * DMatrix and Booster handles are void* opaque types.  We use PHP-managed CData
 * objects for the handle containers (no `false` flag → PHP GC manages lifetime of
 * the CData wrapper).  The XGBoost-internal objects pointed to by those handles are
 * freed explicitly via the C API — FFI::free() is NOT called on handle CData.
 *
 * Hyperparameters
 * ───────────────
 * @param int    $maxDepth    Max tree depth.                     (default 6)
 * @param float  $eta         Learning rate (step shrinkage).     (default 0.3)
 * @param int    $nEstimators Number of boosting rounds.          (default 100)
 * @param string $objective   XGBoost objective string.           (default 'binary:logistic')
 * @param int    $nThreads    CPU threads; 0 = XGBoost default.   (default 0)
 * @param float  $subsample   Row sub-sampling ratio ∈ (0, 1].    (default 1.0)
 * @param float  $colsample   Column sub-sampling per tree ∈ (0, 1]. (default 1.0)
 */
final class XGBoostFFI
{
    // =========================================================================
    // XGBoost C API — singleton FFI instance
    // =========================================================================

    /** @var FFI|null Singleton wrapping libxgboost.so. */
    private static ?FFI $xgb = null;

    /** Candidate library filenames tried in order on load. */
    private const CANDIDATE_LIBS = [
        'libxgboost.so',
        'libxgboost.so.2',
        'libxgboost.so.1',
        'libxgboost.dylib',   // macOS
    ];

    /**
     * Minimal XGBoost C API cdef — only the symbols we actually call.
     *
     * Opaque handles are declared as void* so we never need to know their
     * internal layout.  `bst_ulong` is always unsigned long (64-bit on LP64).
     *
     * Note: `const` qualifiers on pointer parameters are dropped where they
     * cause PHP FFI type-system conflicts (e.g., `float **out_result`).
     */
    private const XGB_HEADER = <<<'XGB'
        typedef void          *DMatrixHandle;
        typedef void          *BoosterHandle;
        typedef unsigned long  bst_ulong;

        /* ── DMatrix lifecycle ───────────────────────────────────────────── */

        /* Create a DMatrix from a row-major float[nrow * ncol] buffer.
         * XGBoost makes its own internal copy of the data — callers may free
         * their buffer immediately after this call returns. */
        int XGDMatrixCreateFromMat(
            const float    *data,
            bst_ulong       nrow,
            bst_ulong       ncol,
            float           missing,
            DMatrixHandle  *out);

        /* Attach a float[len] info array (e.g. "label", "weight") to a DMatrix. */
        int XGDMatrixSetFloatInfo(
            DMatrixHandle   handle,
            const char     *field,
            const float    *array,
            bst_ulong       len);

        int XGDMatrixFree(DMatrixHandle handle);

        int XGDMatrixNumRow(DMatrixHandle handle, bst_ulong *out);
        int XGDMatrixNumCol(DMatrixHandle handle, bst_ulong *out);

        /* ── Booster lifecycle ───────────────────────────────────────────── */

        /* Create a Booster for the given watchlist of DMatrix objects. */
        int XGBoosterCreate(
            const DMatrixHandle *dmats,
            bst_ulong            len,
            BoosterHandle       *out);

        int XGBoosterFree(BoosterHandle handle);

        /* ── Configuration ───────────────────────────────────────────────── */

        int XGBoosterSetParam(
            BoosterHandle  handle,
            const char    *name,
            const char    *value);

        /* ── Training ────────────────────────────────────────────────────── */

        int XGBoosterUpdateOneIter(
            BoosterHandle  handle,
            int            iter,
            DMatrixHandle  dtrain);

        /* ── Prediction ──────────────────────────────────────────────────── */

        /* Returns a pointer to an XGBoost-owned float array — do NOT free.
         * out_len receives the total number of floats returned.
         * For binary objectives: out_len == nrow (one probability per sample).
         * For multi-class softmax: out_len == nrow * num_class. */
        int XGBoosterPredict(
            BoosterHandle  handle,
            DMatrixHandle  dmat,
            int            option_mask,
            unsigned       ntree_limit,
            int            training,
            bst_ulong     *out_len,
            float         **out_result);

        /* ── Error handling ──────────────────────────────────────────────── */

        /* Returns a thread-local error string for the most recent failing call. */
        const char *XGBGetLastError();
    XGB;

    // =========================================================================
    // Instance state
    // =========================================================================

    /**
     * PHP-managed CData holding the training DMatrixHandle (void*).
     * The XGBoost-internal DMatrix is freed via XGDMatrixFree() in freeModel().
     * The CData wrapper is released by PHP GC when set to null.
     *
     * @var FFI\CData|null
     */
    private ?FFI\CData $trainDMatrix = null;

    /**
     * PHP-managed CData holding the BoosterHandle (void*).
     * Freed via XGBoosterFree() in freeModel() BEFORE the DMatrix is freed,
     * because the Booster holds an internal reference to its training DMatrix.
     *
     * @var FFI\CData|null
     */
    private ?FFI\CData $booster = null;

    /** Class labels from training, in insertion order (index 0 first). */
    private array $classes = [];

    /** Whether a model has been trained and is ready for predict(). */
    private bool $trained = false;

    // =========================================================================
    // Constructor / hyperparameters
    // =========================================================================

    public function __construct(
        private int    $maxDepth    = 6,
        private float  $eta         = 0.3,
        private int    $nEstimators = 100,
        private string $objective   = 'binary:logistic',
        private int    $nThreads    = 0,
        private float  $subsample   = 1.0,
        private float  $colsample   = 1.0,
    ) {
        if ($maxDepth < 1) {
            throw new InvalidArgumentException("maxDepth must be >= 1, $maxDepth given.");
        }

        if ($eta <= 0.0 || $eta > 1.0) {
            throw new InvalidArgumentException("eta must be in (0, 1], $eta given.");
        }

        if ($nEstimators < 1) {
            throw new InvalidArgumentException("nEstimators must be >= 1, $nEstimators given.");
        }

        if ($subsample <= 0.0 || $subsample > 1.0) {
            throw new InvalidArgumentException("subsample must be in (0, 1], $subsample given.");
        }

        if ($colsample <= 0.0 || $colsample > 1.0) {
            throw new InvalidArgumentException("colsample must be in (0, 1], $colsample given.");
        }
    }

    // =========================================================================
    // Training
    // =========================================================================

    /**
     * Train an XGBoost model on an ArrowDataset.
     *
     * Pointer hand-off sequence
     * ─────────────────────────
     *   1. getFeaturePointer() → double[rows*cols] (BLAS scope, owned by dataset)
     *   2. Cast double → float into a fresh XGBoost-scope float[rows*cols] temp.
     *   3. XGDMatrixCreateFromMat(temp_float_ptr, …) — XGBoost takes its own copy.
     *   4. FFI::free(temp_float_ptr) — our temp is no longer needed.
     *   5. getLabelPointer() → float[rows] (BLAS scope, owned by dataset)
     *   6. Copy into XGBoost-scope float[rows] temp (cross-scope safety).
     *   7. XGDMatrixSetFloatInfo(dm, "label", temp_label_ptr, rows).
     *   8. FFI::free(temp_label_ptr) — XGBoost has its own copy.
     *   9. XGBoosterCreate, XGBoosterSetParam × N, XGBoosterUpdateOneIter × T.
     *
     * @param ArrowDataset $dataset Labeled dataset with features and labels.
     * @throws InvalidArgumentException If the dataset has no labels.
     * @throws RuntimeException         On any XGBoost C API failure.
     */
    public function train(ArrowDataset $dataset): void
    {
        if (!$dataset->hasLabels()) {
            throw new InvalidArgumentException(
                'ArrowDataset must have labels for supervised training.'
            );
        }

        $xgb  = self::getInstance();
        $rows = $dataset->getRows();
        $cols = $dataset->getCols();

        // ── Step 1: Cast double* features → XGBoost-scope float* ─────────────
        //
        // XGDMatrixCreateFromMat() accepts only const float*, but our ArrowDataset
        // stores features as double[] in the BLAS FFI scope.  We perform one O(n)
        // copy here; allocating in the XGBoost scope avoids any cross-scope pointer
        // type mismatch that PHP FFI might enforce at call time.
        $fltCount  = $rows * $cols;
        $featFltBuf = $xgb->new("float[$fltCount]", false);  // Unmanaged in XGB scope
        $dblPtr    = $dataset->getFeaturePointer();            // double* — DO NOT free

        for ($idx = 0; $idx < $fltCount; ++$idx) {
            $featFltBuf[$idx] = (float) $dblPtr[$idx];
        }

        // ── Step 2: Create XGBoost DMatrix from the float* feature buffer ─────
        //
        // The DMatrix handle is an opaque void* returned via an output parameter.
        // We allocate a PHP-managed DMatrixHandle CData to receive it.
        // FFI::addr() gives the void** the C function needs.
        $dmHandle = $xgb->new('DMatrixHandle');   // PHP-managed void* (initially null)

        $rc = $xgb->XGDMatrixCreateFromMat(
            $featFltBuf,         // const float* data — our temporary feature buffer
            (int) $rows,         // bst_ulong nrow
            (int) $cols,         // bst_ulong ncol
            NAN,                 // float missing — NaN = XGBoost sentinel for "no value"
            FFI::addr($dmHandle) // DMatrixHandle *out — XGBoost writes the handle here
        );

        // ── Immediately free the feature temp buffer ──────────────────────────
        // XGBoost has copied the data into its own internal store.
        // After this point $featFltBuf is freed — do not access it.
        FFI::free($featFltBuf);

        $this->checkRC($rc, 'XGDMatrixCreateFromMat');

        // ── Step 3: Attach labels — copy float* from BLAS scope → XGB scope ──
        //
        // getLabelPointer() returns a BLAS-scope float[rows] holding 0-based
        // float class indices.  We allocate a fresh XGBoost-scope float[rows]
        // so the pointer type is unambiguous to the XGB cdef.
        $lblSrcPtr = $dataset->getLabelPointer();               // float[rows] — DO NOT free
        $lblFltBuf = $xgb->new("float[$rows]", false);          // Unmanaged in XGB scope

        for ($i = 0; $i < $rows; ++$i) {
            $lblFltBuf[$i] = (float) $lblSrcPtr[$i];
        }

        $rc = $xgb->XGDMatrixSetFloatInfo(
            $dmHandle,    // DMatrixHandle — the DMatrix we just created
            'label',      // field name recognised by XGBoost
            $lblFltBuf,   // const float* — our XGBoost-scope label buffer
            (int) $rows   // length
        );

        // ── Free label temp buffer — XGBoost has its own copy now ─────────────
        FFI::free($lblFltBuf);

        $this->checkRC($rc, 'XGDMatrixSetFloatInfo');

        // ── Release any previously trained model before overwriting ───────────
        $this->freeModel();

        // Store the training DMatrix — freed in freeModel() / __destruct()
        $this->trainDMatrix = $dmHandle;
        $this->classes      = $dataset->classes();

        // ── Step 4: Create the Booster ────────────────────────────────────────
        //
        // XGBoosterCreate takes an array of DMatrixHandles (watchlist).
        // We pass a single-element unmanaged array holding the training DMatrix.
        $bstHandle = $xgb->new('BoosterHandle');                 // PHP-managed void*
        $dmList    = $xgb->new('DMatrixHandle[1]', false);       // Unmanaged watchlist

        // Copy the DMatrixHandle void* value into the watchlist array
        $dmList[0] = $dmHandle;

        $rc = $xgb->XGBoosterCreate(
            $dmList,             // const DMatrixHandle* — the watchlist array
            1,                   // bst_ulong len — one DMatrix
            FFI::addr($bstHandle)// BoosterHandle *out — XGBoost writes the handle here
        );

        // ── Watchlist array is only needed for the Create call; free it now ───
        FFI::free($dmList);

        $this->checkRC($rc, 'XGBoosterCreate');

        $this->booster = $bstHandle;

        // ── Step 5: Apply hyperparameters ──────────────────────────────────────
        $params = [
            'max_depth'        => (string) $this->maxDepth,
            'eta'              => (string) $this->eta,
            'objective'        => $this->objective,
            'nthread'          => (string) $this->nThreads,
            'subsample'        => (string) $this->subsample,
            'colsample_bytree' => (string) $this->colsample,
            'eval_metric'      => 'logloss',
        ];

        foreach ($params as $name => $value) {
            $rc = $xgb->XGBoosterSetParam($bstHandle, $name, $value);
            $this->checkRC($rc, "XGBoosterSetParam($name=$value)");
        }

        // ── Step 6: Training loop ─────────────────────────────────────────────
        for ($iter = 0; $iter < $this->nEstimators; ++$iter) {
            $rc = $xgb->XGBoosterUpdateOneIter($bstHandle, $iter, $dmHandle);
            $this->checkRC($rc, "XGBoosterUpdateOneIter(iter=$iter)");
        }

        $this->trained = true;
    }

    // =========================================================================
    // Prediction
    // =========================================================================

    /**
     * Predict class labels for an ArrowDataset.
     *
     * For binary objectives: thresholds at 0.5 (positive class = classes[1]).
     * For multi-class softmax: argmax over the per-class probability vector.
     *
     * @param ArrowDataset $dataset Feature-only dataset (labels not required).
     * @return list<string> Predicted class labels, one per sample.
     * @throws RuntimeException If train() has not been called.
     */
    public function predict(ArrowDataset $dataset): array
    {
        $proba = $this->runPredict($dataset);

        $isBinary   = str_contains($this->objective, 'binary');
        $numClasses = count($this->classes);
        $rows       = $dataset->getRows();
        $predictions = [];

        if ($isBinary) {
            // Binary: one sigmoid probability per sample; threshold at 0.5.
            // classes[0] = negative class, classes[1] = positive class.
            for ($i = 0; $i < $rows; ++$i) {
                $classIdx      = $proba[$i] >= 0.5 ? 1 : 0;
                $predictions[] = $this->classes[$classIdx] ?? (string) $classIdx;
            }
        } else {
            // Multi-class softmax: numClasses probabilities per sample.
            $samplesOut = (int) round(count($proba) / max(1, $numClasses));

            for ($i = 0; $i < $samplesOut; ++$i) {
                $best    = -INF;
                $bestIdx = 0;

                for ($c = 0; $c < $numClasses; ++$c) {
                    $p = $proba[$i * $numClasses + $c];

                    if ($p > $best) {
                        $best    = $p;
                        $bestIdx = $c;
                    }
                }

                $predictions[] = $this->classes[$bestIdx] ?? (string) $bestIdx;
            }
        }

        return $predictions;
    }

    /**
     * Return raw prediction scores without thresholding.
     *
     * For binary objectives: float[rows] of sigmoid probabilities.
     * For multi-class: float[rows * numClasses] in row-major order.
     *
     * @param ArrowDataset $dataset
     * @return list<float>
     */
    public function predictProba(ArrowDataset $dataset): array
    {
        return $this->runPredict($dataset);
    }

    // =========================================================================
    // Memory management
    // =========================================================================

    /**
     * Release the training DMatrix and Booster handles.
     *
     * Called automatically by __destruct() and also before re-training
     * (to prevent the old model from leaking when train() is called again).
     *
     * ── Order matters: free Booster BEFORE DMatrix ────────────────────────────
     * The Booster holds an internal reference to its training DMatrix.
     * Freeing the DMatrix first would leave the Booster with a dangling pointer.
     */
    private function freeModel(): void
    {
        // Use the cached singleton — may be null if getInstance() was never called
        // (e.g., the object was constructed but train() was never reached).
        $xgb = self::$xgb;

        if ($xgb === null) {
            return;
        }

        // ── 1. Free the Booster first ─────────────────────────────────────────
        if ($this->booster !== null) {
            // XGBoosterFree releases XGBoost's internal Booster object.
            // The PHP CData wrapper ($this->booster) is released by GC on null assignment.
            $xgb->XGBoosterFree($this->booster);
            $this->booster = null;
        }

        // ── 2. Free the training DMatrix ─────────────────────────────────────
        if ($this->trainDMatrix !== null) {
            // XGDMatrixFree releases XGBoost's internal DMatrix.
            $xgb->XGDMatrixFree($this->trainDMatrix);
            $this->trainDMatrix = null;
        }

        $this->trained = false;
        $this->classes = [];
    }

    /**
     * Destructor — ensures XGBoost-internal objects are freed when the PHP
     * classifier goes out of scope, even if freeModel() was never called.
     *
     * XGBoost allocates large internal trees; leaking them is fatal in
     * long-running scan pipelines where the classifier is re-trained each cycle.
     */
    public function __destruct()
    {
        $this->freeModel();
    }

    // =========================================================================
    // FFI singleton
    // =========================================================================

    /**
     * Load and return the XGBoost FFI singleton.
     *
     * @throws RuntimeException If libxgboost.so cannot be found or loaded.
     */
    public static function getInstance(): FFI
    {
        if (self::$xgb !== null) {
            return self::$xgb;
        }

        $lastError = null;

        foreach (self::CANDIDATE_LIBS as $lib) {
            try {
                self::$xgb = FFI::cdef(self::XGB_HEADER, $lib);

                return self::$xgb;
            } catch (\Throwable $e) {
                $lastError = $e;
            }
        }

        throw new RuntimeException(
            'Could not load libxgboost. '
            . 'Install XGBoost and ensure libxgboost.so is in LD_LIBRARY_PATH. '
            . 'Last error: ' . ($lastError?->getMessage() ?? 'unknown')
        );
    }

    /**
     * Returns true if libxgboost is available and loadable.
     */
    public static function isAvailable(): bool
    {
        try {
            self::getInstance();

            return true;
        } catch (\Throwable) {
            return false;
        }
    }

    // =========================================================================
    // Internal helpers
    // =========================================================================

    /**
     * Core prediction runner shared by predict() and predictProba().
     *
     * Creates a local DMatrix for the prediction dataset, runs XGBoosterPredict,
     * collects the output float array into a PHP array, and frees the local DMatrix.
     *
     * @param ArrowDataset $dataset
     * @return list<float> Raw XGBoost prediction scores.
     * @throws RuntimeException If not trained or on any C API failure.
     */
    private function runPredict(ArrowDataset $dataset): array
    {
        if (!$this->trained) {
            throw new RuntimeException(
                'XGBoostFFI model has not been trained. Call train() first.'
            );
        }

        $xgb  = self::getInstance();
        $rows = $dataset->getRows();
        $cols = $dataset->getCols();

        // ── Cast double* features → XGBoost-scope float* ─────────────────────
        $fltCount    = $rows * $cols;
        $featFltBuf  = $xgb->new("float[$fltCount]", false);
        $dblPtr      = $dataset->getFeaturePointer();   // double* — DO NOT free

        for ($idx = 0; $idx < $fltCount; ++$idx) {
            $featFltBuf[$idx] = (float) $dblPtr[$idx];
        }

        // ── Build a local predict DMatrix ─────────────────────────────────────
        $predDm = $xgb->new('DMatrixHandle');   // PHP-managed void*

        $rc = $xgb->XGDMatrixCreateFromMat(
            $featFltBuf, (int) $rows, (int) $cols,
            NAN, FFI::addr($predDm)
        );

        // ── Free the feature temp buffer — XGBoost has its own copy ───────────
        FFI::free($featFltBuf);

        $this->checkRC($rc, 'XGDMatrixCreateFromMat (predict)');

        // ── Run XGBoosterPredict ──────────────────────────────────────────────
        //
        // XGBoosterPredict fills:
        //   *out_len    — number of floats in the prediction array
        //   *out_result — pointer to an XGBoost-owned float array (DO NOT free)
        //
        // We use PHP-managed CData for both output containers.
        $outLen    = $xgb->new('bst_ulong');  // Receives the prediction count
        $outResult = $xgb->new('float *');    // Receives the result array pointer

        $rc = $xgb->XGBoosterPredict(
            $this->booster,
            $predDm,
            0,                   // option_mask: 0 = normal prediction output
            0,                   // ntree_limit: 0 = use all trees
            0,                   // training: 0 = inference mode (dropout off)
            FFI::addr($outLen),
            FFI::addr($outResult)
        );

        // ── Free the prediction DMatrix — it was created locally ──────────────
        // IMPORTANT: free BEFORE checkRC so we never leak on error.
        $xgb->XGDMatrixFree($predDm);

        $this->checkRC($rc, 'XGBoosterPredict');

        // ── Read prediction values from XGBoost's internal buffer ─────────────
        //
        // $outResult is a float* CData pointing into XGBoost-owned memory.
        // PHP FFI pointer CData supports array-style subscript access via
        // pointer arithmetic: $outResult[i] reads the float at ptr + i.
        // DO NOT call FFI::free($outResult) — we do not own this memory.
        $count  = (int) $outLen->cdata;
        $scores = [];

        for ($i = 0; $i < $count; ++$i) {
            $scores[] = (float) $outResult[$i];
        }

        return $scores;
    }

    /**
     * Assert a XGBoost C API return code is success (0) and throw on failure.
     *
     * @param int    $rc   Return code from the XGBoost C function.
     * @param string $call Name of the function for the error message.
     * @throws RuntimeException With the XGBoost error string on failure.
     */
    private function checkRC(int $rc, string $call): void
    {
        if ($rc !== 0) {
            $err = self::$xgb !== null
                ? (string) self::$xgb->XGBGetLastError()
                : 'FFI not initialised';

            throw new RuntimeException(
                "XGBoost $call failed (rc=$rc): $err"
            );
        }
    }
}
