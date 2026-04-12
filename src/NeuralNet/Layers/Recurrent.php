<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\Layers;

use Tensor\Matrix;
use Tensor\Vector;
use Rubix\ML\Deferred;
use Rubix\ML\Helpers\Params;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Initializers\Xavier2;
use Rubix\ML\NeuralNet\Initializers\Constant;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Generator;

use function count;
use function array_fill;
use function array_reverse;

/**
 * Recurrent
 *
 * A simple (Elman) RNN hidden layer that processes sequences step-by-step and
 * returns the last hidden state for use in regression or classification heads.
 *
 * **Input layout** (rows = inputSize × seqLen, cols = batchSize):
 *   Rows are grouped by time step: step0_feat0, step0_feat1, …, step1_feat0, …
 *   i.e. row index = timeStep × inputSize + featureIndex
 *
 * **Output**: (hiddenSize, batchSize) — last hidden state h(T).
 *
 * **Recurrence**: h(t) = tanh( Wx · x(t) + Wh · h(t−1) + b )
 *
 * Gradients are computed via Backpropagation Through Time (BPTT) over all
 * time steps, giving exact gradients within a single mini-batch.
 *
 * Clip gradients with `$gradClip` to mitigate exploding-gradient pathology,
 * which is common in vanilla RNNs trained on long sequences.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      The Rubix ML Community
 */
class Recurrent implements Hidden, Parametric
{
    /**
     * Clipping threshold for the L2-norm of parameter gradients (0 = no clip).
     */
    protected const GRAD_CLIP = 5.0;

    /**
     * Number of hidden units.
     *
     * @var positive-int
     */
    protected int $hiddenSize;

    /**
     * Number of input features per time step.
     *
     * @var positive-int
     */
    protected int $inputSize;

    /**
     * Input-to-hidden weights Wx: (hiddenSize × inputSize).
     *
     * @var Parameter|null
     */
    protected ?Parameter $wx = null;

    /**
     * Hidden-to-hidden weights Wh: (hiddenSize × hiddenSize).
     *
     * @var Parameter|null
     */
    protected ?Parameter $wh = null;

    /**
     * Bias vector b: (hiddenSize,).
     *
     * @var Parameter|null
     */
    protected ?Parameter $bias = null;

    /**
     * Inferred sequence length (set during initialize).
     *
     * @var positive-int|null
     */
    protected ?int $seqLen = null;

    /**
     * Hidden states h(0), h(1), …, h(T) from the last forward pass.
     * Each entry is a (hiddenSize × batchSize) Matrix.
     *
     * @var Matrix[]|null
     */
    protected ?array $states = null;

    /**
     * Per-step pre-activation values z(t) = Wx·x(t) + Wh·h(t−1) + b.
     * Each entry is a (hiddenSize × batchSize) Matrix.
     *
     * @var Matrix[]|null
     */
    protected ?array $preacts = null;

    /**
     * Per-step inputs x(t): each a (inputSize × batchSize) Matrix.
     *
     * @var Matrix[]|null
     */
    protected ?array $inputs = null;

    /**
     * @param int $hiddenSize Number of hidden units (≥ 1)
     * @param int $inputSize  Input features per time step (≥ 1)
     * @throws InvalidArgumentException
     */
    public function __construct(int $hiddenSize = 64, int $inputSize = 1)
    {
        if ($hiddenSize < 1) {
            throw new InvalidArgumentException("Hidden size must be"
                . " at least 1, $hiddenSize given.");
        }

        if ($inputSize < 1) {
            throw new InvalidArgumentException("Input size must be"
                . " at least 1, $inputSize given.");
        }

        $this->hiddenSize = $hiddenSize;
        $this->inputSize  = $inputSize;
    }

    /**
     * Return the output width (= hiddenSize).
     *
     * @throws RuntimeException
     * @return positive-int
     */
    public function width() : int
    {
        return $this->hiddenSize;
    }

    /**
     * Initialize weights. fanIn = inputSize × seqLen.
     *
     * @param positive-int $fanIn
     * @throws InvalidArgumentException
     * @return positive-int hiddenSize
     */
    public function initialize(int $fanIn) : int
    {
        if ($fanIn % $this->inputSize !== 0) {
            throw new InvalidArgumentException(
                "fanIn ($fanIn) must be divisible by inputSize ({$this->inputSize})."
            );
        }

        $this->seqLen = $fanIn / $this->inputSize;

        $init = new Xavier2();

        $this->wx   = new Parameter($init->initialize($this->inputSize, $this->hiddenSize));
        $this->wh   = new Parameter($init->initialize($this->hiddenSize, $this->hiddenSize));
        $this->bias = new Parameter(
            (new Constant(0.0))->initialize(1, $this->hiddenSize)->columnAsVector(0)
        );

        return $this->hiddenSize;
    }

    /**
     * Forward pass: process all time steps and return the final hidden state.
     *
     * @param Matrix $input (inputSize × seqLen, batchSize)
     * @throws RuntimeException
     * @return Matrix (hiddenSize, batchSize)
     */
    public function forward(Matrix $input) : Matrix
    {
        if ($this->wx === null || $this->seqLen === null) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        [$states, $preacts, $inputs] = $this->runForward($input, true);

        $this->states  = $states;
        $this->preacts = $preacts;
        $this->inputs  = $inputs;

        return $states[$this->seqLen]; // h(T)
    }

    /**
     * Inference pass (no state caching).
     *
     * @param Matrix $input
     * @throws RuntimeException
     * @return Matrix (hiddenSize, batchSize)
     */
    public function infer(Matrix $input) : Matrix
    {
        if ($this->wx === null || $this->seqLen === null) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        [$states] = $this->runForward($input, false);

        return $states[$this->seqLen];
    }

    /**
     * Backward pass via BPTT.
     *
     * @param Deferred $prevGradient Gradient w.r.t. h(T): (hiddenSize, batchSize)
     * @param Optimizer $optimizer
     * @throws RuntimeException
     * @return Deferred Gradient w.r.t. full input: (inputSize × seqLen, batchSize)
     */
    public function back(Deferred $prevGradient, Optimizer $optimizer) : Deferred
    {
        if ($this->wx === null || $this->seqLen === null) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        if ($this->states === null) {
            throw new RuntimeException('Must perform a forward pass'
                . ' before backpropagating.');
        }

        $dHNext   = $prevGradient(); // (hiddenSize, batchSize)
        $states   = $this->states;
        $preacts  = $this->preacts;
        $inputs   = $this->inputs;
        $T        = $this->seqLen;
        $H        = $this->hiddenSize;
        $I        = $this->inputSize;
        $B        = $dHNext->n();

        $wxArr = $this->wx->param()->asArray();
        $whArr = $this->wh->param()->asArray();

        // Gradient accumulators
        $dWx   = $this->zeroMatrix($H, $I);
        $dWh   = $this->zeroMatrix($H, $H);
        $dBias = $this->zeroVector($H);

        // Full input gradient: (I * T, B)
        $dInputFull = $this->zeroMatrix($I * $T, $B);

        $dHNextArr = $dHNext->asArray();

        for ($t = $T - 1; $t >= 0; $t--) {
            // tanh derivative: 1 − h(t)^2
            $hArr  = $states[$t + 1]->asArray();
            $zArr  = $preacts[$t]->asArray();
            $xtArr = $inputs[$t]->asArray();

            // δ(t) = dH ⊙ (1 − h²)
            $delta = $this->zeroMatrix($H, $B);

            for ($h = 0; $h < $H; $h++) {
                for ($b = 0; $b < $B; $b++) {
                    $delta[$h][$b] = $dHNextArr[$h][$b] * (1.0 - $hArr[$h][$b] ** 2);
                }
            }

            // dWx += δ · x(t)^T
            for ($h = 0; $h < $H; $h++) {
                for ($i = 0; $i < $I; $i++) {
                    $sum = 0.0;

                    for ($b = 0; $b < $B; $b++) {
                        $sum += $delta[$h][$b] * $xtArr[$i][$b];
                    }

                    $dWx[$h][$i] += $sum;
                }
            }

            // dWh += δ · h(t-1)^T
            $hPrevArr = $states[$t]->asArray();

            for ($h = 0; $h < $H; $h++) {
                for ($hh = 0; $hh < $H; $hh++) {
                    $sum = 0.0;

                    for ($b = 0; $b < $B; $b++) {
                        $sum += $delta[$h][$b] * $hPrevArr[$hh][$b];
                    }

                    $dWh[$h][$hh] += $sum;
                }
            }

            // dBias += Σ_b δ
            for ($h = 0; $h < $H; $h++) {
                $s = 0.0;

                for ($b = 0; $b < $B; $b++) {
                    $s += $delta[$h][$b];
                }

                $dBias[$h] += $s;
            }

            // dInput(t) = Wx^T · δ — stored in full input gradient
            for ($i = 0; $i < $I; $i++) {
                $rowIdx = $t * $I + $i;

                for ($b = 0; $b < $B; $b++) {
                    $val = 0.0;

                    for ($h = 0; $h < $H; $h++) {
                        $val += $wxArr[$h][$i] * $delta[$h][$b];
                    }

                    $dInputFull[$rowIdx][$b] = $val;
                }
            }

            // Propagate gradient to previous time step: dH = Wh^T · δ
            $newDH = $this->zeroMatrix($H, $B);

            for ($h = 0; $h < $H; $h++) {
                for ($b = 0; $b < $B; $b++) {
                    $val = 0.0;

                    for ($hh = 0; $hh < $H; $hh++) {
                        $val += $whArr[$hh][$h] * $delta[$hh][$b];
                    }

                    $newDH[$h][$b] = $val;
                }
            }

            $dHNextArr = $newDH;
        }

        // Clip gradients
        $dWxMat  = $this->clipGrad(Matrix::quick($dWx));
        $dWhMat  = $this->clipGrad(Matrix::quick($dWh));
        $dBVec   = $this->clipVec(Vector::quick($dBias));
        $dInpMat = Matrix::quick($dInputFull);

        $this->wx->update($dWxMat, $optimizer);
        $this->wh->update($dWhMat, $optimizer);
        $this->bias->update($dBVec, $optimizer);

        $this->states  = null;
        $this->preacts = null;
        $this->inputs  = null;

        return new Deferred(static fn () => $dInpMat);
    }

    /**
     * Yield trainable parameters.
     *
     * @throws RuntimeException
     * @return Generator<Parameter>
     */
    public function parameters() : Generator
    {
        if ($this->wx === null) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        yield 'wx'   => $this->wx;
        yield 'wh'   => $this->wh;
        yield 'bias' => $this->bias;
    }

    /**
     * Restore parameters from a Snapshot.
     *
     * @param Parameter[] $parameters
     */
    public function restore(array $parameters) : void
    {
        $this->wx   = $parameters['wx'];
        $this->wh   = $parameters['wh'];
        $this->bias = $parameters['bias'];
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Recurrent ('
            . Params::stringify([
                'hidden size' => $this->hiddenSize,
                'input size'  => $this->inputSize,
            ])
            . ')';
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /**
     * Run the forward pass through all time steps.
     *
     * @param Matrix $input   (I*T, B)
     * @param bool   $cache   Whether to cache intermediate states (training)
     * @return array{Matrix[], Matrix[], Matrix[]}  [states, preacts, inputs]
     */
    protected function runForward(Matrix $input, bool $cache) : array
    {
        $T         = $this->seqLen;
        $H         = $this->hiddenSize;
        $I         = $this->inputSize;
        $B         = $input->n();
        $inputArr  = $input->asArray();
        $wxArr     = $this->wx->param()->asArray();
        $whArr     = $this->wh->param()->asArray();
        $biasArr   = $this->bias->param()->asArray();

        // h(0) = zeros
        $hArr = $this->zeroMatrix($H, $B);

        $states  = [Matrix::quick($hArr)];
        $preacts = [];
        $inputs  = [];

        for ($t = 0; $t < $T; $t++) {
            // Extract x(t): rows t*I … (t+1)*I-1
            $xt = $this->zeroMatrix($I, $B);

            for ($i = 0; $i < $I; $i++) {
                $xt[$i] = $inputArr[$t * $I + $i];
            }

            // z = Wx · x + Wh · h + b
            $z = $this->zeroMatrix($H, $B);

            for ($h = 0; $h < $H; $h++) {
                for ($b = 0; $b < $B; $b++) {
                    $val = $biasArr[$h];

                    for ($i = 0; $i < $I; $i++) {
                        $val += $wxArr[$h][$i] * $xt[$i][$b];
                    }

                    for ($hh = 0; $hh < $H; $hh++) {
                        $val += $whArr[$h][$hh] * $hArr[$hh][$b];
                    }

                    $z[$h][$b] = $val;
                }
            }

            // h(t) = tanh(z)
            $hNew = $this->zeroMatrix($H, $B);

            for ($h = 0; $h < $H; $h++) {
                for ($b = 0; $b < $B; $b++) {
                    $hNew[$h][$b] = tanh($z[$h][$b]);
                }
            }

            $hArr = $hNew;

            $states[] = Matrix::quick($hNew);

            if ($cache) {
                $preacts[] = Matrix::quick($z);
                $inputs[]  = Matrix::quick($xt);
            }
        }

        return [$states, $preacts, $inputs];
    }

    /**
     * Allocate a zero-filled 2-D array (rows × cols).
     *
     * @return float[][]
     */
    protected function zeroMatrix(int $rows, int $cols) : array
    {
        $row = array_fill(0, $cols, 0.0);
        return array_fill(0, $rows, $row);
    }

    /**
     * Allocate a zero-filled 1-D array (length $n).
     *
     * @return float[]
     */
    protected function zeroVector(int $n) : array
    {
        return array_fill(0, $n, 0.0);
    }

    /**
     * Clip a Matrix gradient by L∞ element-wise norm.
     */
    protected function clipGrad(Matrix $g) : Matrix
    {
        $c = self::GRAD_CLIP;

        return $g->map(fn ($v) => max(-$c, min($c, $v)));
    }

    /**
     * Clip a Vector gradient.
     */
    protected function clipVec(Vector $v) : Vector
    {
        $c = self::GRAD_CLIP;

        return $v->map(fn ($x) => max(-$c, min($c, $x)));
    }
}
