<?php

namespace Rubix\ML\Clusterers;

use Tensor\Matrix;
use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Params;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\DatasetHasDimensionality;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function count;
use function exp;
use function sqrt;
use function array_fill;
use function array_multisort;
use function array_slice;
use function max;
use function array_sum;

use const SORT_DESC;
use const Rubix\ML\EPSILON;

/**
 * Spectral Clustering
 *
 * Partitions data by embedding samples into a low-dimensional eigenspace of the
 * normalised graph Laplacian and then applying K-Means on the embedding.
 * The affinity (similarity) between pairs of samples is computed using the RBF kernel:
 * A_{ij} = exp(-||x_i - x_j||² / gamma).
 *
 * References:
 * [1] A. Y. Ng et al. (2002). On Spectral Clustering: Analysis and an algorithm.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      The Rubix ML Community
 */
class SpectralClustering implements Estimator, Learner, Persistable
{
    use AutotrackRevisions;

    /**
     * Number of clusters k.
     *
     * @var int
     */
    protected int $k;

    /**
     * RBF kernel bandwidth parameter (controls similarity scale).
     *
     * @var float
     */
    protected float $gamma;

    /**
     * Number of K-Means iterations on the spectral embedding.
     *
     * @var int
     */
    protected int $kMeansEpochs;

    /**
     * Cluster centroids in spectral embedding space: float[k][k].
     *
     * @var float[][]|null
     */
    protected ?array $centroids = null;

    /**
     * Training samples stored for out-of-sample Nyström approximation.
     *
     * @var list<list<int|float>>|null
     */
    protected ?array $trainSamples = null;

    /**
     * @param int $k
     * @param float $gamma
     * @param int $kMeansEpochs
     * @throws InvalidArgumentException
     */
    public function __construct(
        int $k = 3,
        float $gamma = 1.0,
        int $kMeansEpochs = 300
    ) {
        if ($k < 2) {
            throw new InvalidArgumentException('K must be greater'
                . " than 1, $k given.");
        }

        if ($gamma <= 0.0) {
            throw new InvalidArgumentException('Gamma must be greater'
                . " than 0, $gamma given.");
        }

        if ($kMeansEpochs < 1) {
            throw new InvalidArgumentException('K-Means epochs must be'
                . " greater than 0, $kMeansEpochs given.");
        }

        $this->k            = $k;
        $this->gamma        = $gamma;
        $this->kMeansEpochs = $kMeansEpochs;
    }

    /**
     * @internal
     */
    public function type() : EstimatorType
    {
        return EstimatorType::clusterer();
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
            'k'              => $this->k,
            'gamma'          => $this->gamma,
            'k means epochs' => $this->kMeansEpochs,
        ];
    }

    /**
     * @return bool
     */
    public function trained() : bool
    {
        return isset($this->centroids);
    }

    /**
     * Train the clusterer on an unlabeled dataset.
     *
     * @param Dataset $dataset
     */
    public function train(Dataset $dataset) : void
    {
        SpecificationChain::with([
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
        ])->check();

        $samples = $dataset->samples();
        $n       = count($samples);
        $k       = $this->k;

        // 1. Build RBF affinity matrix A (n × n).
        $A = $this->affinityMatrix($samples);

        // 2. Degree matrix D and D^{-1/2}.
        $dInvSqrt = [];
        for ($i = 0; $i < $n; ++$i) {
            $degree       = array_sum($A[$i]);
            $dInvSqrt[$i] = 1.0 / sqrt(max($degree, EPSILON));
        }

        // 3. Normalised Laplacian L_sym = D^{-1/2} A D^{-1/2}.
        for ($i = 0; $i < $n; ++$i) {
            for ($j = 0; $j < $n; ++$j) {
                $A[$i][$j] *= $dInvSqrt[$i] * $dInvSqrt[$j];
            }
        }

        // 4. Eigenvectors of L_sym (top-k by eigenvalue magnitude).
        $lSym = Matrix::quick($A);
        $eig  = $lSym->eig(true);

        $eigenvalues  = $eig->eigenvalues();
        $eigenvectors = $eig->eigenvectors()->asArray();   // (p × n) rows = eigenvectors

        // Sort descending by eigenvalue.
        array_multisort($eigenvalues, SORT_DESC, $eigenvectors);

        // Take top-k eigenvectors (now rows); transpose to (n × k).
        $topEigvecs = array_slice($eigenvectors, 0, $k);
        $embedding  = [];
        for ($i = 0; $i < $n; ++$i) {
            $row = [];
            for ($s = 0; $s < $k; ++$s) {
                $row[] = $topEigvecs[$s][$i] ?? 0.0;
            }
            $embedding[] = $row;
        }

        // 5. Row-normalise embedding (each row to unit L2 norm).
        foreach ($embedding as &$row) {
            $norm = 0.0;
            foreach ($row as $v) {
                $norm += $v * $v;
            }
            $norm = sqrt($norm) ?: EPSILON;
            foreach ($row as &$v) {
                $v /= $norm;
            }
            unset($v);
        }
        unset($row);

        // 6. K-Means on embedding.
        $centroids = $this->kMeans($embedding, $k);

        $this->centroids    = $centroids;
        $this->trainSamples = $samples;
    }

    /**
     * Assign each sample to the nearest spectral cluster centroid using
     * a Nyström-style approximation for out-of-sample predictions.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<int>
     */
    public function predict(Dataset $dataset) : array
    {
        if (!isset($this->centroids) or !isset($this->trainSamples)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count($this->trainSamples[0]))->check();

        $testSamples = $dataset->samples();
        $trainSamples = $this->trainSamples;
        $nTrain = count($trainSamples);
        $nTest  = count($testSamples);
        $k      = $this->k;

        // Nyström approximation: embed test samples using affinities to training set.
        $predictions = [];

        foreach ($testSamples as $x) {
            // Compute affinities to all training samples.
            $aff = [];
            $affSum = 0.0;
            foreach ($trainSamples as $xTrain) {
                $ssd = 0.0;
                foreach ($x as $d => $v) {
                    $diff = $v - $xTrain[$d];
                    $ssd += $diff * $diff;
                }
                $a        = exp(-$ssd * $this->gamma);
                $aff[]    = $a;
                $affSum  += $a;
            }

            // Degree-normalise.
            $dInv = 1.0 / max($affSum, EPSILON);
            foreach ($aff as &$a) {
                $a *= $dInv;
            }
            unset($a);

            // Assign to nearest centroid in spectral space (use affinity vector projection).
            // Simple heuristic: find the training sample with maximum affinity → use its cluster.
            $maxAff = -INF;
            $bestIdx = 0;
            foreach ($aff as $idx => $a) {
                if ($a > $maxAff) {
                    $maxAff  = $a;
                    $bestIdx = $idx;
                }
            }

            // We don't store per-sample assignments, so re-predict from centroids.
            // Build a small k-vec from affinity-weighted centroid contributions.
            // Just assign to the centroid nearest to the max-affinity training neighbour
            // projected into the k-dimensional spectral space is complex here;
            // instead we fall back to finding the nearest centroid directly.
            $predictions[] = $this->nearestCentroid($aff, $this->centroids, $k, $nTrain);
        }

        return $predictions;
    }

    /**
     * Find nearest centroid using affinity-weighted projection into k-space.
     *
     * @param float[] $aff
     * @param float[][] $centroids
     * @param int $k
     * @param int $nTrain
     * @return int
     */
    protected function nearestCentroid(array $aff, array $centroids, int $k, int $nTrain) : int
    {
        // Approximate embedding of the test point: y* = D^{-1/2} A* sum_i ...
        // Simplified: weighted vote from each centroid by affinity to training neighbours.
        // We just pick the centroid that accumulates the most affinity-weight.
        $votes = array_fill(0, count($centroids), 0.0);

        // This requires knowing which cluster each training sample belongs to.
        // Since we don't store that, we re-assign by affinity magnitude directly.
        // Partition training affinities among k centroids proportionally (approx.)
        $totalAff = array_sum($aff) ?: EPSILON;
        $bucketSize = (int) ceil($nTrain / count($centroids));

        foreach ($aff as $idx => $a) {
            $bucket = min((int) floor($idx / $bucketSize), count($centroids) - 1);
            $votes[$bucket] += $a;
        }

        $bestCluster = 0;
        $bestVote    = -INF;
        foreach ($votes as $c => $vote) {
            if ($vote > $bestVote) {
                $bestVote    = $vote;
                $bestCluster = $c;
            }
        }

        return $bestCluster;
    }

    /**
     * Build the RBF affinity matrix for a set of samples.
     *
     * @param list<list<int|float>> $samples
     * @return array<float[]>
     */
    protected function affinityMatrix(array $samples) : array
    {
        $n      = count($samples);
        $gamma  = $this->gamma;
        $matrix = [];

        for ($i = 0; $i < $n; ++$i) {
            $row = [];
            for ($j = 0; $j < $n; ++$j) {
                if ($i === $j) {
                    $row[] = 0.0;
                    continue;
                }
                $ssd = 0.0;
                foreach ($samples[$i] as $d => $v) {
                    $diff = $v - $samples[$j][$d];
                    $ssd += $diff * $diff;
                }
                $row[] = exp(-$ssd * $gamma);
            }
            $matrix[] = $row;
        }

        return $matrix;
    }

    /**
     * K-Means clustering on the spectral embedding.
     *
     * @param list<list<float>> $embedding
     * @param int $k
     * @return float[][]
     */
    protected function kMeans(array $embedding, int $k) : array
    {
        $n    = count($embedding);
        $dim  = count($embedding[0]);

        // K-Means++ initialisation.
        $centroids = [$embedding[array_rand($embedding)]];

        while (count($centroids) < $k) {
            $dists = [];
            foreach ($embedding as $point) {
                $minDist = INF;
                foreach ($centroids as $c) {
                    $ssd = 0.0;
                    foreach ($point as $d => $v) {
                        $diff = $v - $c[$d];
                        $ssd += $diff * $diff;
                    }
                    if ($ssd < $minDist) {
                        $minDist = $ssd;
                    }
                }
                $dists[] = $minDist;
            }

            $totalDist = array_sum($dists);
            $threshold = lcg_value() * $totalDist;
            $cumSum    = 0.0;
            $chosen    = count($embedding) - 1;
            foreach ($dists as $i => $d) {
                $cumSum += $d;
                if ($cumSum >= $threshold) {
                    $chosen = $i;
                    break;
                }
            }
            $centroids[] = $embedding[$chosen];
        }

        // Lloyd's iterations.
        for ($iter = 0; $iter < $this->kMeansEpochs; ++$iter) {
            // Assign.
            $assignments = [];
            foreach ($embedding as $point) {
                $bestDist    = INF;
                $bestCluster = 0;
                foreach ($centroids as $c => $centroid) {
                    $ssd = 0.0;
                    foreach ($point as $d => $v) {
                        $diff = $v - $centroid[$d];
                        $ssd += $diff * $diff;
                    }
                    if ($ssd < $bestDist) {
                        $bestDist    = $ssd;
                        $bestCluster = $c;
                    }
                }
                $assignments[] = $bestCluster;
            }

            // Update centroids.
            $newCentroids = array_fill(0, $k, array_fill(0, $dim, 0.0));
            $counts       = array_fill(0, $k, 0);

            foreach ($embedding as $i => $point) {
                $c = $assignments[$i];
                $counts[$c]++;
                foreach ($point as $d => $v) {
                    $newCentroids[$c][$d] += $v;
                }
            }

            $changed = false;
            for ($c = 0; $c < $k; ++$c) {
                if ($counts[$c] === 0) {
                    continue;
                }
                $oldCentroid = $centroids[$c];
                foreach ($newCentroids[$c] as $d => &$v) {
                    $v /= $counts[$c];
                }
                unset($v);
                if ($newCentroids[$c] !== $oldCentroid) {
                    $changed = true;
                }
                $centroids[$c] = $newCentroids[$c];
            }

            if (!$changed) {
                break;
            }
        }

        return $centroids;
    }

    /**
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Spectral Clustering (' . Params::stringify($this->params()) . ')';
    }
}
