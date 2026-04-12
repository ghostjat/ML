<?php

declare(strict_types=1);

namespace Rubix\ML\Experiments;

use Rubix\ML\DataType;
use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Probabilistic;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Verbose;
use Rubix\ML\Traits\LoggerAware;
use InvalidArgumentException;
use RuntimeException;

use const INF;
use function count;
use function range;
use function array_fill;
use function array_flip;
use function array_slice;
use function array_values;
use function array_unique;
use function exp;
use function log;
use function max;
use function min;
use function round;
use function floor;
use function is_nan;
use function mt_rand;

class GradientBoostClassifier implements Estimator, Learner, Probabilistic, Verbose
{
    use LoggerAware;
    
    protected int   $estimators;
    protected float $learningRate;
    protected int   $maxDepth;
    protected float $minChildWeight;
    protected float $lambda;
    protected float $gamma;
    protected float $colsampleBytree;
    protected float $subsample;
    protected float $maxDeltaStep;

    protected array $trees   = [];
    protected array $classes = [];
    protected bool  $isMulticlass = false;

    protected float $f0Binary = 0.0;
    protected array $f0Multi  = [];
    protected array $classWeights = [];
    protected array $binEdges = []; 

    public function __construct(
        int   $estimators      = 200,
        float $learningRate    = 0.05,
        int   $maxDepth        = 5,
        float $minChildWeight  = 1.0,
        float $lambda          = 1.0,
        float $gamma           = 0.0,
        float $colsampleBytree = 1.0,
        float $subsample       = 1.0,
        float $maxDeltaStep    = 0.0
    ) {
        $this->estimators      = $estimators;
        $this->learningRate    = $learningRate;
        $this->maxDepth        = $maxDepth;
        $this->minChildWeight  = $minChildWeight;
        $this->lambda          = $lambda;
        $this->gamma           = $gamma;
        $this->colsampleBytree = $colsampleBytree;
        $this->subsample       = min(1.0, max(0.0, $subsample));
        $this->maxDeltaStep    = max(0.0, $maxDeltaStep);
    }

    public function type(): EstimatorType
    {
        return EstimatorType::classifier();
    }

    public function train(Dataset $dataset): void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('Dataset must be an instance of Labeled.');
        }

        $samples = array_values($dataset->samples());
        $labels  = array_values($dataset->labels());
        $n       = count($samples);

        $this->classes = array_values(array_unique($labels));
        $nClasses      = count($this->classes);

        if ($nClasses < 2) {
            throw new InvalidArgumentException('Need at least 2 classes to train a classifier.');
        }

        $this->isMulticlass = $nClasses > 2;
        $this->trees        = [];

        // Automatic inverse-frequency class weights
        $classCounts = array_count_values($labels);
        foreach ($this->classes as $c) {
            $count = max(1, $classCounts[$c] ?? 1);
            $this->classWeights[$c] = $n / ($nClasses * $count);
        }

        // PRE-COMPUTATION: Discretize features into integer histograms (Bins 0-255)
        $numFeatures = count($samples[0]);
        $columns = [];
        $binnedSamples = array_fill(0, $n, []);
        $this->binEdges = [];

        for ($j = 0; $j < $numFeatures; $j++) {
            $min = INF;
            $max = -INF;
            foreach ($samples as $i => $row) {
                $val = $row[$j];
                if ($val !== null && !is_nan((float)$val)) {
                    $v = (float)$val;
                    if ($v < $min) $min = $v;
                    if ($v > $max) $max = $v;
                }
            }

            if ($min === INF) { // Edge case: entire column is missing
                $min = 0.0;
                $max = 1.0;
            }

            $binSize = max(1e-8, ($max - $min) / 255.0);
            $this->binEdges[$j] = ['min' => $min, 'binSize' => $binSize];

            $colBinned = array_fill(0, $n, 256);
            for ($i = 0; $i < $n; $i++) {
                $val = $samples[$i][$j];
                if ($val === null || is_nan((float)$val)) {
                    $bin = 256;
                } else {
                    $bin = (int) floor(((float)$val - $min) / $binSize);
                    $bin = max(0, min(255, $bin));
                }
                $colBinned[$i] = $bin;
                $binnedSamples[$i][$j] = $bin; // Save locally for tree updates
            }
            $columns[$j] = $colBinned;
        }

        $allIndices = range(0, $n - 1);

        if ($this->isMulticlass) {
            $this->trainMulticlass($binnedSamples, $columns, $allIndices, $n, $labels);
        } else {
            $this->trainBinary($binnedSamples, $columns, $allIndices, $n, $labels);
        }
    }

    protected function trainBinary(
        array  $binnedSamples,
        array &$columns,
        array  $allIndices,
        int    $n,
        array  $labels
    ): void {
        $class1 = $this->classes[1];

        $yNum = array_fill(0, $n, 0.0);
        $weights = array_fill(0, $n, 1.0);
        
        $posCount = 0;
        for ($i = 0; $i < $n; $i++) {
            $yNum[$i] = $labels[$i] === $class1 ? 1.0 : 0.0;
            $weights[$i] = $this->classWeights[$labels[$i]];
            if ($yNum[$i] === 1.0) ++$posCount;
        }

        $pBar           = max(1e-9, min(1.0 - 1e-9, $posCount / $n));
        $f0             = log($pBar / (1.0 - $pBar));
        $this->f0Binary = $f0;

        $F    = array_fill(0, $n, $f0);
        $grad = array_fill(0, $n, 0.0);
        $hess = array_fill(0, $n, 0.0);

        $lr = $this->learningRate;

        for ($m = 0; $m < $this->estimators; ++$m) {
            $epochLoss = 0.0;
            for ($i = 0; $i < $n; ++$i) {
                $s = $F[$i];
                if ($s >= 0.0) {
                    $p = 1.0 / (1.0 + exp(-$s));
                } else {
                    $e = exp($s);
                    $p = $e / (1.0 + $e);
                }
                $epochLoss -= $yNum[$i] * log($p + 1e-15) + (1.0 - $yNum[$i]) * log(1.0 - $p + 1e-15);
                $w = $weights[$i];
                $grad[$i] = $w * ($p - $yNum[$i]);
                $hess[$i] = $w * max($p * (1.0 - $p), 1e-16);
            }
            if ($this->logger) {
                $this->logger->info("Epoch " . ($m + 1) . " - Binary Log Loss: " . round($epochLoss / $n, 6));
            }
            $treeIndices = $this->drawSubsample($allIndices, $n);

            $tree = new NewtonTree(
                $this->maxDepth, $this->minChildWeight, $this->lambda, $this->gamma, $this->colsampleBytree, $this->maxDeltaStep
            );
            $tree->train($columns, $grad, $hess, $treeIndices);

            $this->trees[] = $tree;

            for ($i = 0; $i < $n; $i++) {
                $F[$i] += $lr * $tree->predictSample($binnedSamples[$i]);
            }
        }
    }

    protected function trainMulticlass(
            array $binnedSamples,
            array &$columns,
            array $allIndices,
            int $n,
            array $labels
    ): void {
        $k = count($this->classes);
        $classIndex = array_flip($this->classes);

        $yInt = array_fill(0, $n, 0);
        $weights = array_fill(0, $n, 1.0);
        $classCounts = array_fill(0, $k, 0);

        for ($i = 0; $i < $n; ++$i) {
            $cIdx = $classIndex[$labels[$i]];
            $yInt[$i] = $cIdx;
            $weights[$i] = $this->classWeights[$labels[$i]] ?? 1.0;
            ++$classCounts[$cIdx];
        }

        $this->f0Multi = array_fill(0, $k, 0.0);

        $F = [];
        for ($i = 0; $i < $n; $i++) {
            $F[$i] = $this->f0Multi;
        }

        $lr = $this->learningRate;
        $totalWeight = array_sum($weights); // <--- HERE IS THE INITIALIZATION!

        // Pre-allocate Epoch Snapshot Matrices (JIT flat-array optimization)
        $epochG = [];
        $epochH = [];
        for ($c = 0; $c < $k; ++$c) {
            $epochG[$c] = array_fill(0, $n, 0.0);
            $epochH[$c] = array_fill(0, $n, 0.0);
        }

        for ($m = 0; $m < $this->estimators; ++$m) {
            $epochLoss = 0.0;
            
            // MATHEMATICAL FIX: Snapshot Phase.
            // Calculate true probabilities, gradients, and Hessians for ALL classes simultaneously.
            for ($i = 0; $i < $n; $i++) {
                $Fi = $F[$i];
                $maxF = $Fi[0];
                for ($c = 1; $c < $k; $c++) {
                    if ($Fi[$c] > $maxF) {
                        $maxF = $Fi[$c];
                    }
                }

                $sumExp = 0.0;
                $exps = [];
                for ($c = 0; $c < $k; $c++) {
                    $e = exp($Fi[$c] - $maxF);
                    $exps[$c] = $e;
                    $sumExp += $e;
                }

                $invSum = 1.0 / max($sumExp, 1e-12);

                $trueClassIndex = $yInt[$i];
                $pTrue = $exps[$trueClassIndex] * $invSum;
                $epochLoss -= $weights[$i] * log($pTrue + 1e-15);//$epochLoss -= log($pTrue + 1e-15);

                for ($c = 0; $c < $k; $c++) {
                    $p = $exps[$c] * $invSum;
                    $w = $weights[$i];

                    // True Multiclass Softmax Gradient: g_c = p_c - y_c
                    $rawGradient = $p - ($yInt[$i] === $c ? 1.0 : 0.0);
                    
                    // MATHEMATICAL FIX: Gradient Clipping
                    // Constrains extreme signal noise in financial data to calibrate probabilities.
                    $clippedGradient = max(-1.0, min(1.0, $rawGradient));

                    // MATHEMATICAL FIX: True Diagonal Hessian Approximation
                    // H_cc = p_c * (1 - p_c)
                    $h = $p * (1.0 - $p);

                    $epochG[$c][$i] = $w * $clippedGradient;
                    $epochH[$c][$i] = $w * max($h, 1e-16);
                }
            }

            if ($this->logger) {
                $this->logger->info("Epoch " . ($m + 1) . " - Weighted Cross-Entropy Loss: " . round($epochLoss / $totalWeight, 6));
            }
            $roundTrees = [];
            
            // MATHEMATICAL FIX: Draw the subsample ONCE per epoch.
            // ALL classes must train on the exact same rows to keep Softmax logits synchronized.
            $epochSubsampleIndices = $this->drawSubsample($allIndices, $n);

            for ($c = 0; $c < $k; $c++) {
                $tree = new NewtonTree(
                    $this->maxDepth, $this->minChildWeight, $this->lambda, $this->gamma, $this->colsampleBytree, $this->maxDeltaStep
                );
                
                // Pass the synchronized subsample to all trees
                $tree->train($columns, $epochG[$c], $epochH[$c], $epochSubsampleIndices);
                $roundTrees[$c] = $tree;
            }

            $this->trees[] = $roundTrees;

            for ($c = 0; $c < $k; $c++) {
                $tree = $roundTrees[$c];
                for ($i = 0; $i < $n; $i++) {
                    $F[$i][$c] += $lr * $tree->predictSample($binnedSamples[$i]);
                }
            }
        }
    }

    protected function drawSubsample(array $allIndices, int $n): array
    {
        if ($this->subsample >= 1.0) {
            return $allIndices;
        }

        $size = max(2, (int) round($n * $this->subsample));
        $pool = $allIndices;

        for ($i = 0; $i < $size; $i++) {
            $j        = mt_rand($i, $n - 1);
            $tmp      = $pool[$i];
            $pool[$i] = $pool[$j];
            $pool[$j] = $tmp;
        }

        return array_slice($pool, 0, $size);
    }

    private function getBinnedSamples(array $samples): array
    {
        $binned = [];
        foreach ($samples as $i => $row) {
            $binnedRow = [];
            foreach ($row as $j => $val) {
                if ($val === null || is_nan((float)$val)) {
                    $binnedRow[$j] = 256;
                } else {
                    $edge = $this->binEdges[$j];
                    $bin = (int) floor(((float)$val - $edge['min']) / $edge['binSize']);
                    $binnedRow[$j] = max(0, min(255, $bin));
                }
            }
            $binned[] = $binnedRow;
        }
        return $binned;
    }

    public function predict(Dataset $dataset): array
    {
        if (empty($this->trees)) throw new RuntimeException('Estimator has not been trained.');

        $samples = array_values($dataset->samples());
        $binnedSamples = $this->getBinnedSamples($samples);

        return $this->isMulticlass
            ? $this->predictMulticlass($binnedSamples)
            : $this->predictBinary($binnedSamples);
    }

    protected function predictBinary(array $binnedSamples): array
    {
        $n  = count($binnedSamples);
        $lr = $this->learningRate;
        $F  = array_fill(0, $n, $this->f0Binary);

        foreach ($this->trees as $tree) {
            for ($i = 0; $i < $n; $i++) {
                $F[$i] += $lr * $tree->predictSample($binnedSamples[$i]);
            }
        }

        $c0   = $this->classes[0];
        $c1   = $this->classes[1];
        $pred = [];

        for ($i = 0; $i < $n; $i++) {
            $pred[] = $F[$i] >= 0.0 ? $c1 : $c0;
        }

        return $pred;
    }

    protected function predictMulticlass(array $binnedSamples): array
    {
        $n  = count($binnedSamples);
        $k  = count($this->classes);
        $lr = $this->learningRate;
        $f0 = $this->f0Multi ?: array_fill(0, $k, 0.0);

        $F = [];
        for ($i = 0; $i < $n; $i++) {
            $F[$i] = $f0;
        }

        foreach ($this->trees as $round) {
            for ($c = 0; $c < $k; $c++) {
                $tree = $round[$c];
                for ($i = 0; $i < $n; $i++) {
                    $F[$i][$c] += $lr * $tree->predictSample($binnedSamples[$i]);
                }
            }
        }

        $classes = $this->classes;
        $pred    = [];

        for ($i = 0; $i < $n; $i++) {
            $Fi   = $F[$i];
            $best = 0;
            $max  = $Fi[0];

            for ($c = 1; $c < $k; $c++) {
                if ($Fi[$c] > $max) {
                    $max  = $Fi[$c];
                    $best = $c;
                }
            }

            $pred[] = $classes[$best];
        }

        return $pred;
    }

    public function proba(Dataset $dataset): array
    {
        if (empty($this->trees)) throw new RuntimeException('Estimator has not been trained.');

        $samples = array_values($dataset->samples());
        $binnedSamples = $this->getBinnedSamples($samples);

        return $this->isMulticlass
            ? $this->probaMulticlass($binnedSamples)
            : $this->probaBinary($binnedSamples);
    }

    protected function probaBinary(array $binnedSamples): array
    {
        $n  = count($binnedSamples);
        $lr = $this->learningRate;
        $F  = array_fill(0, $n, $this->f0Binary);

        foreach ($this->trees as $tree) {
            for ($i = 0; $i < $n; $i++) {
                $F[$i] += $lr * $tree->predictSample($binnedSamples[$i]);
            }
        }

        $c0  = $this->classes[0];
        $c1  = $this->classes[1];
        $out = [];

        for ($i = 0; $i < $n; $i++) {
            $s = $F[$i];
            if ($s >= 0.0) {
                $p = 1.0 / (1.0 + exp(-$s));
            } else {
                $e = exp($s);
                $p = $e / (1.0 + $e);
            }
            $out[] = [$c0 => 1.0 - $p, $c1 => $p];
        }

        return $out;
    }

    protected function probaMulticlass(array $binnedSamples): array
    {
        $n  = count($binnedSamples);
        $k  = count($this->classes);
        $lr = $this->learningRate;
        $f0 = $this->f0Multi ?: array_fill(0, $k, 0.0);

        $F = [];
        for ($i = 0; $i < $n; $i++) {
            $F[$i] = $f0;
        }

        foreach ($this->trees as $round) {
            for ($c = 0; $c < $k; $c++) {
                $tree = $round[$c];
                for ($i = 0; $i < $n; $i++) {
                    $F[$i][$c] += $lr * $tree->predictSample($binnedSamples[$i]);
                }
            }
        }

        $classes = $this->classes;
        $out     = [];

        for ($i = 0; $i < $n; $i++) {
            $Fi   = $F[$i];
            $maxF = $Fi[0];
            for ($c = 1; $c < $k; $c++) {
                if ($Fi[$c] > $maxF) {
                    $maxF = $Fi[$c];
                }
            }

            $sumExp = 0.0;
            $exps   = array_fill(0, $k, 0.0);
            for ($c = 0; $c < $k; $c++) {
                $e        = exp($Fi[$c] - $maxF);
                $exps[$c] = $e;
                $sumExp  += $e;
            }

            $invSum = 1.0 / max($sumExp, 1e-12);
            $dist   = [];
            for ($c = 0; $c < $k; $c++) {
                $dist[$classes[$c]] = $exps[$c] * $invSum;
            }

            $out[] = $dist;
        }

        return $out;
    }

    public function trained(): bool
    {
        return !empty($this->trees);
    }

    public function compatibility(): array
    {
        return [DataType::continuous()];
    }

    public function params(): array
    {
        return [
            'estimators'       => $this->estimators,
            'learning_rate'    => $this->learningRate,
            'max_depth'        => $this->maxDepth,
            'min_child_weight' => $this->minChildWeight,
            'lambda'           => $this->lambda,
            'gamma'            => $this->gamma,
            'colsample_bytree' => $this->colsampleBytree,
            'subsample'        => $this->subsample,
            'max_delta_step'   => $this->maxDeltaStep,
        ];
    }

    public function __toString(): string {
        return 'Gradient Boost Classifier'
                . ' (estimators: ' . $this->estimators
                . ', lr: ' . $this->learningRate
                . ', max_depth: ' . $this->maxDepth
                . ', lambda: ' . $this->lambda
                . ', subsample: ' . $this->subsample . ')';
    }
}
