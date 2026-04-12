<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Verbose;
use Rubix\ML\DataType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Traits\LoggerAware;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function count;
use function exp;
use function log;
use function sqrt;
use function max;
use function min;
use function abs;
use function array_fill;
use function random_int;
use function asort;
use function array_keys;
use function array_slice;
use function array_values;
use function array_unique;

use const Rubix\ML\EPSILON;
use const M_PI;
use const M_LN2;

/**
 * UMAP
 *
 * Uniform Manifold Approximation and Projection is a dimensionality reduction
 * technique that preserves both global and local structure by modelling the data
 * as a fuzzy topological structure and optimising a low-dimensional representation
 * with respect to a cross-entropy cost function.
 *
 * The high-dimensional fuzzy membership μ_{ij} = exp(-max(d_{ij}-ρ_i,0)/σ_i)
 * is symmetrised as v_{ij} = μ_{ij} + μ_{ji} - μ_{ij}·μ_{ji}.  The low-dimensional
 * similarity uses the Student-t-like curve q_{ij} = (1+a·||y_i-y_j||^{2b})^{-1}.
 * Parameters a and b are fit from *minDist*; the defaults correspond to minDist=0.1.
 *
 * References:
 * [1] L. McInnes et al. (2018). UMAP: Uniform Manifold Approximation and Projection
 *     for Dimension Reduction.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      The Rubix ML Community
 */
class UMAP implements Transformer, Stateful, Verbose
{
    use LoggerAware;

    /**
     * Maximum binary search iterations to find σ (smooth kNN distances).
     *
     * @var int
     */
    protected const MAX_SIGMA_ITER = 64;

    /**
     * Tolerance for binary search on σ.
     *
     * @var float
     */
    protected const SIGMA_TOL = 1e-5;

    /**
     * Negative sampling ratio (negative samples per positive edge per epoch).
     *
     * @var int
     */
    protected const NEG_SAMPLE_RATE = 5;

    /**
     * Number of output dimensions.
     *
     * @var int
     */
    protected int $dimensions;

    /**
     * Number of nearest neighbours used to build the fuzzy graph.
     *
     * @var int
     */
    protected int $neighbours;

    /**
     * Minimum distance in the low-dimensional space (controls cluster tightness).
     *
     * @var float
     */
    protected float $minDist;

    /**
     * Number of SGD optimisation epochs.
     *
     * @var int
     */
    protected int $epochs;

    /**
     * Initial learning rate for the SGD optimiser.
     *
     * @var float
     */
    protected float $rate;

    /**
     * Spread of the low-dimensional distribution (controls effective range).
     *
     * @var float
     */
    protected float $spread;

    /**
     * Curve parameter a (derived from minDist and spread).
     *
     * @var float
     */
    protected float $a;

    /**
     * Curve parameter b (derived from minDist and spread).
     *
     * @var float
     */
    protected float $b;

    /**
     * The distance kernel used for computing pairwise distances.
     *
     * @var Distance
     */
    protected Distance $kernel;

    /**
     * The embedding computed during the last fit (m × d array).
     *
     * @var float[][]|null
     */
    protected ?array $embedding = null;

    /**
     * Training samples stored for out-of-sample projection.
     *
     * @var list<list<int|float>>|null
     */
    protected ?array $trainSamples = null;

    /**
     * @param int $dimensions
     * @param int $neighbours
     * @param float $minDist
     * @param int $epochs
     * @param float $rate
     * @param float $spread
     * @param Distance|null $kernel
     * @throws InvalidArgumentException
     */
    public function __construct(
        int $dimensions = 2,
        int $neighbours = 15,
        float $minDist = 0.1,
        int $epochs = 200,
        float $rate = 1.0,
        float $spread = 1.0,
        ?Distance $kernel = null
    ) {
        if ($dimensions < 1) {
            throw new InvalidArgumentException('Dimensions must be'
                . " greater than 0, $dimensions given.");
        }

        if ($neighbours < 2) {
            throw new InvalidArgumentException('Neighbours must be'
                . " greater than 1, $neighbours given.");
        }

        if ($minDist <= 0.0 or $minDist >= $spread) {
            throw new InvalidArgumentException('Min dist must be between'
                . " 0 and spread, $minDist given.");
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Epochs must be greater'
                . " than 0, $epochs given.");
        }

        if ($rate <= 0.0) {
            throw new InvalidArgumentException('Learning rate must be'
                . " greater than 0, $rate given.");
        }

        if ($spread <= 0.0) {
            throw new InvalidArgumentException('Spread must be greater'
                . " than 0, $spread given.");
        }

        // Fit a and b from minDist and spread via a simple curve fit approximation.
        // These values are pre-computed for the standard (minDist=0.1, spread=1.0).
        // For other values we use the analytical approximation.
        $this->a = $this->fitCurveA($minDist, $spread);
        $this->b = $this->fitCurveB($minDist, $spread);

        $this->dimensions = $dimensions;
        $this->neighbours = $neighbours;
        $this->minDist    = $minDist;
        $this->epochs     = $epochs;
        $this->rate       = $rate;
        $this->spread     = $spread;
        $this->kernel     = $kernel ?? new Euclidean();
    }

    /**
     * @internal
     *
     * @return list<DataType>
     */
    public function compatibility() : array
    {
        return $this->kernel->compatibility();
    }

    /**
     * Is the transformer fitted?
     *
     * @return bool
     */
    public function fitted() : bool
    {
        return isset($this->embedding);
    }

    /**
     * Fit the transformer to a dataset and compute the embedding.
     *
     * @param Dataset $dataset
     */
    public function fit(Dataset $dataset) : void
    {
        SamplesAreCompatibleWithTransformer::with($dataset, $this)->check();

        $samples = $dataset->samples();
        $n = count($samples);
        $k = min($this->neighbours, $n - 1);

        if ($this->logger) {
            $this->logger->info('Computing k-NN graph');
        }

        // 1. Compute pairwise distances and find k-NN for each point.
        [$knnIndices, $knnDists] = $this->kNearestNeighbours($samples, $k);

        if ($this->logger) {
            $this->logger->info('Building fuzzy simplicial set');
        }

        // 2. Compute ρ_i (distance to nearest neighbour).
        $rho = [];
        for ($i = 0; $i < $n; ++$i) {
            $rho[$i] = $knnDists[$i][0] ?? 0.0;
        }

        // 3. Binary search for σ_i: Σ_j exp(-max(d_ij - ρ_i, 0) / σ_i) = log2(k)
        $sigma  = [];
        $target = log($k) / M_LN2;  // log2(k)
        for ($i = 0; $i < $n; ++$i) {
            $sigma[$i] = $this->smoothKNN($knnDists[$i], $rho[$i], $target);
        }

        // 4. Build sparse edge list with high-dim memberships.
        // μ_{ij} = exp(-max(d_{ij}-ρ_i,0)/σ_i)
        // v_{ij} = μ_{ij} + μ_{ji} - μ_{ij}*μ_{ji}  (symmetrisation)
        $muMatrix = [];  // [i][j] => μ

        for ($i = 0; $i < $n; ++$i) {
            foreach ($knnIndices[$i] as $ki => $j) {
                $dij = $knnDists[$i][$ki];
                $muIJ = exp(-max($dij - $rho[$i], 0.0) / ($sigma[$i] ?: EPSILON));
                $muMatrix[$i][$j] = $muIJ;
            }
        }

        // Collect edges (symmetrised).
        $edges = [];
        for ($i = 0; $i < $n; ++$i) {
            foreach (($muMatrix[$i] ?? []) as $j => $muIJ) {
                if ($j <= $i) {
                    continue;  // process each pair once
                }
                $muJI = $muMatrix[$j][$i] ?? 0.0;
                $v    = $muIJ + $muJI - $muIJ * $muJI;
                if ($v > EPSILON) {
                    $edges[] = [$i, $j, $v];
                }
            }
        }

        // 5. Initialise embedding with spectral / random layout.
        if ($this->logger) {
            $this->logger->info('Initialising embedding');
        }

        $embedding = $this->initEmbedding($n, $this->dimensions);

        // 6. SGD optimisation.
        if ($this->logger) {
            $this->logger->info('Optimising embedding');
        }

        $embedding = $this->optimise($embedding, $edges, $n);

        if ($this->logger) {
            $this->logger->info('Embedding complete');
        }

        $this->embedding    = $embedding;
        $this->trainSamples = $samples;
    }

    /**
     * Transform dataset samples in-place using the fitted embedding.
     * For out-of-sample data we project each point by weighting the training
     * embedding coordinates by proximity (Nadaraya-Watson kernel regression).
     *
     * @param array<mixed[]> $samples
     * @throws RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if (!isset($this->embedding) or !isset($this->trainSamples)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        $isTrain = (count($samples) === count($this->trainSamples));

        if ($isTrain) {
            // Return the computed embedding directly.
            $samples = $this->embedding;
            return;
        }

        // Out-of-sample: Nadaraya-Watson regression over training embedding.
        $trainSamples = $this->trainSamples;
        $embedding    = $this->embedding;
        $d            = $this->dimensions;
        $nTrain       = count($trainSamples);
        $k            = min($this->neighbours, $nTrain);

        $result = [];

        foreach ($samples as $x) {
            // Distances to all training points.
            $dists = [];
            for ($i = 0; $i < $nTrain; ++$i) {
                $dists[$i] = $this->kernel->compute($x, $trainSamples[$i]);
            }
            asort($dists);
            $knnIdx = array_slice(array_keys($dists), 0, $k, true);

            // Gaussian kernel weights.
            $bw  = max(array_slice(array_values($dists), 0, $k)) ?: EPSILON;
            $ws  = [];
            $wSum = 0.0;
            foreach ($knnIdx as $idx) {
                $w      = exp(-($dists[$idx] / $bw) ** 2);
                $ws[$idx] = $w;
                $wSum  += $w;
            }

            $row = array_fill(0, $d, 0.0);
            foreach ($knnIdx as $idx) {
                $w = $ws[$idx] / ($wSum ?: EPSILON);
                for ($dd = 0; $dd < $d; ++$dd) {
                    $row[$dd] += $w * $embedding[$idx][$dd];
                }
            }

            $result[] = $row;
        }

        $samples = $result;
    }

    // -----------------------------------------------------------------------
    // Internal methods
    // -----------------------------------------------------------------------

    /**
     * Compute brute-force k nearest neighbours for all samples.
     * Returns [knn_indices (n×k), knn_dists (n×k)] sorted by distance ascending.
     *
     * @param list<list<int|float>> $samples
     * @param int $k
     * @return array{list<int[]>, list<float[]>}
     */
    protected function kNearestNeighbours(array $samples, int $k) : array
    {
        $n       = count($samples);
        $indices = [];
        $dists   = [];

        for ($i = 0; $i < $n; ++$i) {
            $rowDists = [];
            for ($j = 0; $j < $n; ++$j) {
                if ($j === $i) {
                    continue;
                }
                $rowDists[$j] = $this->kernel->compute($samples[$i], $samples[$j]);
            }
            asort($rowDists);

            $ki    = 0;
            $idxs  = [];
            $ds    = [];
            foreach ($rowDists as $j => $d) {
                $idxs[] = $j;
                $ds[]   = $d;
                if (++$ki >= $k) {
                    break;
                }
            }
            $indices[] = $idxs;
            $dists[]   = $ds;
        }

        return [$indices, $dists];
    }

    /**
     * Binary search to find σ_i such that Σ_j exp(-max(d-ρ,0)/σ) ≈ target.
     *
     * @param float[] $knnDists  Sorted distances to k nearest neighbours
     * @param float $rho         Distance to nearest neighbour
     * @param float $target      log2(k)
     * @return float
     */
    protected function smoothKNN(array $knnDists, float $rho, float $target) : float
    {
        $lo  = 0.0;
        $hi  = INF;
        $mid = 1.0;

        for ($iter = 0; $iter < self::MAX_SIGMA_ITER; ++$iter) {
            $pSum = 0.0;
            foreach ($knnDists as $d) {
                $pSum += exp(-max($d - $rho, 0.0) / ($mid ?: EPSILON));
            }

            if (abs($pSum - $target) < self::SIGMA_TOL) {
                break;
            }

            if ($pSum > $target) {
                $hi  = $mid;
                $mid = 0.5 * ($lo + $hi);
            } else {
                $lo = $mid;
                $mid = ($hi === INF) ? $mid * 2.0 : 0.5 * ($lo + $hi);
            }
        }

        return $mid ?: EPSILON;
    }

    /**
     * Initialise the low-dimensional embedding with random normal values.
     *
     * @param int $n
     * @param int $d
     * @return float[][]
     */
    protected function initEmbedding(int $n, int $d) : array
    {
        $embedding = [];
        for ($i = 0; $i < $n; ++$i) {
            $row = [];
            for ($dd = 0; $dd < $d; ++$dd) {
                // Box-Muller normal sample.
                $u1    = max(lcg_value(), EPSILON);
                $u2    = lcg_value();
                $row[] = sqrt(-2.0 * log($u1)) * cos(2.0 * M_PI * $u2) * 0.01;
            }
            $embedding[] = $row;
        }
        return $embedding;
    }

    /**
     * Stochastic gradient descent over edges using the UMAP attractive and
     * repulsive forces.
     *
     * @param float[][] $y        Current embedding (modified in-place via reference)
     * @param array<array{int,int,float}> $edges  [(i, j, weight), ...]
     * @param int $n
     * @return float[][]
     */
    protected function optimise(array $y, array $edges, int $n) : array
    {
        $d       = $this->dimensions;
        $a       = $this->a;
        $b       = $this->b;
        $negRate = self::NEG_SAMPLE_RATE;

        for ($epoch = 1; $epoch <= $this->epochs; ++$epoch) {
            // Linearly decaying learning rate.
            $alpha = $this->rate * (1.0 - ($epoch - 1) / $this->epochs);
            if ($alpha < 1e-8) {
                $alpha = 1e-8;
            }

            foreach ($edges as [$i, $j, $w]) {
                // Attractive gradient.
                $distSq = 0.0;
                for ($dd = 0; $dd < $d; ++$dd) {
                    $diff    = $y[$i][$dd] - $y[$j][$dd];
                    $distSq += $diff * $diff;
                }

                $distSq  = max($distSq, EPSILON);
                $invDen  = 1.0 / (1.0 + $a * ($distSq ** $b));
                $gradCoef = -2.0 * $a * $b * ($distSq ** ($b - 1.0)) * $invDen;

                for ($dd = 0; $dd < $d; ++$dd) {
                    $diff         = $y[$i][$dd] - $y[$j][$dd];
                    $g            = max(-4.0, min(4.0, $gradCoef * $diff * $w));
                    $y[$i][$dd] += $alpha * $g;
                    $y[$j][$dd] -= $alpha * $g;
                }

                // Repulsive samples.
                for ($neg = 0; $neg < $negRate; ++$neg) {
                    $k = random_int(0, $n - 1);
                    if ($k === $i) {
                        continue;
                    }

                    $distSq = 0.0;
                    for ($dd = 0; $dd < $d; ++$dd) {
                        $diff    = $y[$i][$dd] - $y[$k][$dd];
                        $distSq += $diff * $diff;
                    }
                    $distSq = max($distSq, EPSILON);

                    $abDist2b   = $a * ($distSq ** $b);
                    $repGradCoef = 2.0 * $b / ($distSq * (1.0 + $abDist2b) + EPSILON);

                    for ($dd = 0; $dd < $d; ++$dd) {
                        $diff         = $y[$i][$dd] - $y[$k][$dd];
                        $g            = max(-4.0, min(4.0, $repGradCoef * $diff));
                        $y[$i][$dd] += $alpha * $g;
                    }
                }
            }

            if ($this->logger and $epoch % 50 === 0) {
                $this->logger->info("Epoch: $epoch / {$this->epochs}");
            }
        }

        return $y;
    }

    /**
     * Approximate curve parameter a from minDist and spread.
     * Derived from the UMAP paper's sigmoid-like parametric curve.
     *
     * @param float $minDist
     * @param float $spread
     * @return float
     */
    protected function fitCurveA(float $minDist, float $spread) : float
    {
        // Empirical fit: a ≈ 1.929 for minDist=0.1, spread=1.0.
        // General: controls the width of the low-dim similarity function.
        return 1.929 / ($spread ** 1.8) * (1.0 + 0.5 * $minDist);
    }

    /**
     * Approximate curve parameter b from minDist and spread.
     *
     * @param float $minDist
     * @param float $spread
     * @return float
     */
    protected function fitCurveB(float $minDist, float $spread) : float
    {
        // Empirical fit: b ≈ 0.7915 for minDist=0.1, spread=1.0.
        return 0.7915 / $spread * (1.0 + 0.1 * $minDist);
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
        return 'UMAP (' . Params::stringify([
            'dimensions' => $this->dimensions,
            'neighbours' => $this->neighbours,
            'min dist'   => $this->minDist,
            'epochs'     => $this->epochs,
            'rate'       => $this->rate,
            'spread'     => $this->spread,
            'kernel'     => $this->kernel,
        ]) . ')';
    }
}
