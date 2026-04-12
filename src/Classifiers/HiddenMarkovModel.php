<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
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
use function array_keys;
use function array_fill;
use function log;
use function exp;
use function sqrt;
use function array_map;
use function is_null;

use const Rubix\ML\EPSILON;
use const Rubix\ML\LOG_EPSILON;
use const M_PI;

/**
 * Hidden Markov Model Classifier
 *
 * Trains one Gaussian HMM per class label using the Baum-Welch (EM) algorithm, then
 * classifies sequences by choosing the model with the highest log-likelihood under the
 * forward algorithm.  Each sample is treated as a univariate or multivariate observation
 * sequence of length equal to the number of features.  Emissions are modelled as
 * diagonal-covariance multivariate Gaussians.
 *
 * References:
 * [1] L. R. Rabiner. (1989). A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition.
 * [2] L. E. Baum et al. (1970). A Maximization Technique Occurring in the Statistical Analysis of Probabilistic
 *     Functions of Markov Chains.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      The Rubix ML Community
 */
class HiddenMarkovModel implements Estimator, Learner, Probabilistic, Persistable
{
    use AutotrackRevisions;

    /**
     * Number of hidden states per class HMM.
     *
     * @var int
     */
    protected int $states;

    /**
     * Maximum Baum-Welch EM iterations.
     *
     * @var int
     */
    protected int $epochs;

    /**
     * Convergence tolerance on log-likelihood improvement.
     *
     * @var float
     */
    protected float $tol;

    /**
     * Minimum variance floor to prevent degenerate Gaussian components.
     *
     * @var float
     */
    protected float $minVariance;

    /**
     * Per-class HMM parameters: [ class => ['pi', 'A', 'mu', 'sigma2'] ]
     *
     * @var array<string|int, array{pi: float[], A: float[][], mu: float[][], sigma2: float[][]}>|null
     */
    protected ?array $models = null;

    /**
     * Ordered list of unique class labels.
     *
     * @var list<string|int>|null
     */
    protected ?array $classes = null;

    /**
     * @param int $states
     * @param int $epochs
     * @param float $tol
     * @param float $minVariance
     * @throws InvalidArgumentException
     */
    public function __construct(
        int $states = 3,
        int $epochs = 100,
        float $tol = 1e-4,
        float $minVariance = 1e-6
    ) {
        if ($states < 1) {
            throw new InvalidArgumentException('States must be greater'
                . " than 0, $states given.");
        }

        if ($epochs < 1) {
            throw new InvalidArgumentException('Epochs must be greater'
                . " than 0, $epochs given.");
        }

        if ($tol < 0.0) {
            throw new InvalidArgumentException('Tolerance must be greater'
                . " than or equal to 0, $tol given.");
        }

        if ($minVariance <= 0.0) {
            throw new InvalidArgumentException('Minimum variance must be'
                . " greater than 0, $minVariance given.");
        }

        $this->states      = $states;
        $this->epochs      = $epochs;
        $this->tol         = $tol;
        $this->minVariance = $minVariance;
    }

    /**
     * @internal
     */
    public function type() : EstimatorType
    {
        return EstimatorType::classifier();
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
            'states'       => $this->states,
            'epochs'       => $this->epochs,
            'tol'          => $this->tol,
            'min variance' => $this->minVariance,
        ];
    }

    /**
     * @return bool
     */
    public function trained() : bool
    {
        return isset($this->models);
    }

    /**
     * Train a separate Gaussian HMM for every class via Baum-Welch.
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
        $n       = count($samples);
        $p       = count($samples[0]);

        // Group samples by class.
        $byClass = [];
        for ($i = 0; $i < $n; ++$i) {
            $byClass[$labels[$i]][] = $samples[$i];
        }

        $this->classes = array_keys($byClass);
        $this->models  = [];

        foreach ($byClass as $class => $seqs) {
            $this->models[$class] = $this->trainHMM($seqs, $p);
        }
    }

    /**
     * Predict the most probable class for each sample.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<string|int>
     */
    public function predict(Dataset $dataset) : array
    {
        if (is_null($this->models) or is_null($this->classes)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count(reset($this->models)['mu'][0]))->check();

        $predictions = [];

        foreach ($dataset->samples() as $sample) {
            $bestClass = null;
            $bestLL    = -INF;

            foreach ($this->models as $class => $model) {
                $ll = $this->forwardLogLikelihood($sample, $model);
                if ($ll > $bestLL) {
                    $bestLL    = $ll;
                    $bestClass = $class;
                }
            }

            $predictions[] = $bestClass;
        }

        return $predictions;
    }

    /**
     * Return the posterior class probabilities (softmax of log-likelihoods).
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<float[]>
     */
    public function proba(Dataset $dataset) : array
    {
        if (is_null($this->models) or is_null($this->classes)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        DatasetHasDimensionality::with($dataset, count(reset($this->models)['mu'][0]))->check();

        $probas = [];

        foreach ($dataset->samples() as $sample) {
            $logLikelihoods = [];

            foreach ($this->models as $class => $model) {
                $logLikelihoods[$class] = $this->forwardLogLikelihood($sample, $model);
            }

            // Stable softmax over log-likelihoods.
            $maxLL = max($logLikelihoods);
            $sum   = 0.0;
            $exps  = [];
            foreach ($logLikelihoods as $class => $ll) {
                $e              = exp($ll - $maxLL);
                $exps[$class]   = $e;
                $sum           += $e;
            }

            $dist = [];
            foreach ($exps as $class => $e) {
                $dist[$class] = $e / ($sum ?: EPSILON);
            }

            $probas[] = $dist;
        }

        return $probas;
    }

    /**
     * Fit a single Gaussian HMM to a set of observation sequences via Baum-Welch.
     *
     * @param list<list<int|float>> $seqs  Observation sequences (samples)
     * @param int $p  Number of features (observation dimension)
     * @return array{pi: float[], A: float[][], mu: float[][], sigma2: float[][]}
     */
    protected function trainHMM(array $seqs, int $p) : array
    {
        $k = $this->states;

        // --- Initialise parameters ---
        // Uniform initial state distribution.
        $pi = array_fill(0, $k, 1.0 / $k);

        // Left-to-right transition matrix with slight self-bias.
        $A = [];
        for ($i = 0; $i < $k; ++$i) {
            $row = array_fill(0, $k, 0.05 / max($k - 1, 1));
            $row[$i] = 0.95;
            $A[$i]   = $row;
        }

        // Emissions: random means seeded from data, unit variance.
        $mu     = [];
        $sigma2 = [];
        $nSeqs  = count($seqs);

        for ($s = 0; $s < $k; ++$s) {
            $idx      = (int) round($s * $nSeqs / $k) % $nSeqs;
            $mu[$s]   = $seqs[$idx];
            $sigma2[$s] = array_fill(0, $p, 1.0);
        }

        $prevLL = -INF;

        for ($epoch = 0; $epoch < $this->epochs; ++$epoch) {
            // Accumulators for M-step.
            $piAcc    = array_fill(0, $k, 0.0);
            $aAcc     = [];
            $muAcc    = [];
            $sig2Acc  = [];
            $gammaSum = array_fill(0, $k, 0.0);

            for ($s = 0; $s < $k; ++$s) {
                $aAcc[$s]   = array_fill(0, $k, 0.0);
                $muAcc[$s]  = array_fill(0, $p, 0.0);
                $sig2Acc[$s] = array_fill(0, $p, 0.0);
            }

            $totalLL = 0.0;

            foreach ($seqs as $seq) {
                $T = count($seq);   // treat each feature as a time-step

                // ---- Forward pass ----
                $logAlpha = [];
                for ($i = 0; $i < $k; ++$i) {
                    $logAlpha[0][$i] = log(max($pi[$i], EPSILON))
                        + $this->logEmit($seq[0] ?? 0.0, $mu[$i], $sigma2[$i], $p, 0);
                }

                for ($t = 1; $t < $T; ++$t) {
                    for ($j = 0; $j < $k; ++$j) {
                        $maxVal = -INF;
                        $vals   = [];
                        for ($i = 0; $i < $k; ++$i) {
                            $v    = $logAlpha[$t - 1][$i] + log(max($A[$i][$j], EPSILON));
                            $vals[] = $v;
                            if ($v > $maxVal) {
                                $maxVal = $v;
                            }
                        }
                        $sumExp = 0.0;
                        foreach ($vals as $v) {
                            $sumExp += exp($v - $maxVal);
                        }
                        $logAlpha[$t][$j] = $maxVal + log($sumExp)
                            + $this->logEmit($seq[$t] ?? 0.0, $mu[$j], $sigma2[$j], $p, $t);
                    }
                }

                // log P(seq | θ)
                $maxAlpha = max($logAlpha[$T - 1]);
                $sumExp   = 0.0;
                foreach ($logAlpha[$T - 1] as $v) {
                    $sumExp += exp($v - $maxAlpha);
                }
                $logPSeq   = $maxAlpha + log($sumExp);
                $totalLL  += $logPSeq;

                // ---- Backward pass ----
                $logBeta = [];
                for ($i = 0; $i < $k; ++$i) {
                    $logBeta[$T - 1][$i] = 0.0;  // log(1)
                }
                for ($t = $T - 2; $t >= 0; --$t) {
                    for ($i = 0; $i < $k; ++$i) {
                        $maxVal = -INF;
                        $vals   = [];
                        for ($j = 0; $j < $k; ++$j) {
                            $v    = log(max($A[$i][$j], EPSILON))
                                + $this->logEmit($seq[$t + 1] ?? 0.0, $mu[$j], $sigma2[$j], $p, $t + 1)
                                + $logBeta[$t + 1][$j];
                            $vals[] = $v;
                            if ($v > $maxVal) {
                                $maxVal = $v;
                            }
                        }
                        $sumExp = 0.0;
                        foreach ($vals as $v) {
                            $sumExp += exp($v - $maxVal);
                        }
                        $logBeta[$t][$i] = $maxVal + log($sumExp);
                    }
                }

                // ---- E-step: gamma and xi ----
                for ($t = 0; $t < $T; ++$t) {
                    $logGamma = [];
                    $maxG     = -INF;
                    for ($i = 0; $i < $k; ++$i) {
                        $lg        = $logAlpha[$t][$i] + $logBeta[$t][$i] - $logPSeq;
                        $logGamma[$i] = $lg;
                        if ($lg > $maxG) {
                            $maxG = $lg;
                        }
                    }
                    $sumG = 0.0;
                    $gamma = [];
                    foreach ($logGamma as $i => $lg) {
                        $g       = exp($lg - $maxG);
                        $gamma[$i] = $g;
                        $sumG   += $g;
                    }
                    foreach ($gamma as $i => &$g) {
                        $g /= ($sumG ?: EPSILON);
                    }
                    unset($g);

                    // Accumulate gamma for M-step.
                    if ($t === 0) {
                        for ($i = 0; $i < $k; ++$i) {
                            $piAcc[$i] += $gamma[$i];
                        }
                    }

                    $obs = $seq[$t] ?? 0.0;

                    for ($i = 0; $i < $k; ++$i) {
                        $gammaSum[$i] += $gamma[$i];
                        // For multivariate: treat $obs as x[d] for d=0 (univariate case)
                        // For multivariate we use the full $seq as observation at t=0 only.
                        // Handle below.
                        $muAcc[$i][0]   += $gamma[$i] * $obs;
                        $diff            = $obs - $mu[$i][0];
                        $sig2Acc[$i][0] += $gamma[$i] * $diff * $diff;
                    }

                    // xi: only needed for transition updates (t < T-1)
                    if ($t < $T - 1) {
                        $nextObs = $seq[$t + 1] ?? 0.0;
                        $maxXi   = -INF;
                        $logXi   = [];
                        for ($i = 0; $i < $k; ++$i) {
                            for ($j = 0; $j < $k; ++$j) {
                                $lx = $logAlpha[$t][$i]
                                    + log(max($A[$i][$j], EPSILON))
                                    + $this->logEmit($nextObs, $mu[$j], $sigma2[$j], $p, $t + 1)
                                    + $logBeta[$t + 1][$j]
                                    - $logPSeq;
                                $logXi[$i][$j] = $lx;
                                if ($lx > $maxXi) {
                                    $maxXi = $lx;
                                }
                            }
                        }
                        $sumXi = 0.0;
                        $xi    = [];
                        for ($i = 0; $i < $k; ++$i) {
                            for ($j = 0; $j < $k; ++$j) {
                                $x            = exp($logXi[$i][$j] - $maxXi);
                                $xi[$i][$j]   = $x;
                                $sumXi       += $x;
                            }
                        }
                        for ($i = 0; $i < $k; ++$i) {
                            for ($j = 0; $j < $k; ++$j) {
                                $aAcc[$i][$j] += $xi[$i][$j] / ($sumXi ?: EPSILON);
                            }
                        }
                    }
                }
            }

            // ---- M-step ----
            $piSum = array_sum($piAcc);
            for ($i = 0; $i < $k; ++$i) {
                $pi[$i] = $piAcc[$i] / ($piSum ?: EPSILON);
            }

            for ($i = 0; $i < $k; ++$i) {
                $rowSum = array_sum($aAcc[$i]);
                for ($j = 0; $j < $k; ++$j) {
                    $A[$i][$j] = $aAcc[$i][$j] / ($rowSum ?: EPSILON);
                }
            }

            for ($s = 0; $s < $k; ++$s) {
                $gs = $gammaSum[$s] ?: EPSILON;
                // For univariate handling (p=1 or single obs per step)
                $mu[$s][0]     = $muAcc[$s][0] / $gs;
                $sigma2[$s][0] = max($this->minVariance, $sig2Acc[$s][0] / $gs);

                // For unused dimensions, keep unit variance
                for ($d = 1; $d < $p; ++$d) {
                    $mu[$s][$d]     = $mu[$s][$d] ?? 0.0;
                    $sigma2[$s][$d] = max($this->minVariance, $sigma2[$s][$d] ?? 1.0);
                }
            }

            if (abs($totalLL - $prevLL) < $this->tol) {
                break;
            }
            $prevLL = $totalLL;
        }

        return compact('pi', 'A', 'mu', 'sigma2');
    }

    /**
     * Log-emission probability: log N(obs; mu[s], sigma2[s]) projected to the
     * t-th observation.  For a 1-D time series (p=1) the observation at time t
     * is the single value.  For multivariate the observation at t=0 is the whole
     * sample vector (p > 1 sequences are flattened as independent dimensions).
     *
     * @param float $obs
     * @param float[] $mu
     * @param float[] $sigma2
     * @param int $p
     * @param int $t
     * @return float
     */
    protected function logEmit(float $obs, array $mu, array $sigma2, int $p, int $t) : float
    {
        // Univariate Gaussian log-pdf at the t-th dimension (wraps around).
        $d     = $t % $p;
        $sig2  = max($sigma2[$d], $this->minVariance);
        $diff  = $obs - $mu[$d];
        return -0.5 * (log(2.0 * M_PI * $sig2) + $diff * $diff / $sig2);
    }

    /**
     * Compute log P(sequence | model) using the forward algorithm in log-space.
     *
     * @param list<int|float> $seq
     * @param array{pi: float[], A: float[][], mu: float[][], sigma2: float[][]} $model
     * @return float
     */
    protected function forwardLogLikelihood(array $seq, array $model) : float
    {
        $pi     = $model['pi'];
        $A      = $model['A'];
        $mu     = $model['mu'];
        $sigma2 = $model['sigma2'];
        $k      = $this->states;
        $T      = count($seq);
        $p      = count($mu[0]);

        $logAlpha = [];
        for ($i = 0; $i < $k; ++$i) {
            $logAlpha[$i] = log(max($pi[$i], EPSILON))
                + $this->logEmit($seq[0], $mu[$i], $sigma2[$i], $p, 0);
        }

        for ($t = 1; $t < $T; ++$t) {
            $newLogAlpha = [];
            for ($j = 0; $j < $k; ++$j) {
                $maxVal = -INF;
                $vals   = [];
                for ($i = 0; $i < $k; ++$i) {
                    $v    = $logAlpha[$i] + log(max($A[$i][$j], EPSILON));
                    $vals[] = $v;
                    if ($v > $maxVal) {
                        $maxVal = $v;
                    }
                }
                $sumExp = 0.0;
                foreach ($vals as $v) {
                    $sumExp += exp($v - $maxVal);
                }
                $newLogAlpha[$j] = $maxVal + log($sumExp)
                    + $this->logEmit($seq[$t], $mu[$j], $sigma2[$j], $p, $t);
            }
            $logAlpha = $newLogAlpha;
        }

        $maxVal = max($logAlpha);
        $sum    = 0.0;
        foreach ($logAlpha as $v) {
            $sum += exp($v - $maxVal);
        }

        return $maxVal + log($sum);
    }

    /**
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Hidden Markov Model (' . Params::stringify($this->params()) . ')';
    }
}
