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
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Generator;

use function count;
use function array_fill;

/**
 * LSTM
 *
 * A Long Short-Term Memory hidden layer that mitigates the vanishing-gradient
 * problem of vanilla RNNs through a gated cell-state mechanism.
 *
 * **Equations** (all operations are element-wise unless noted):
 *
 *   f(t) = σ( Wf · [h(t−1), x(t)] + bf )   forget gate
 *   i(t) = σ( Wi · [h(t−1), x(t)] + bi )   input gate
 *   g(t) = tanh( Wg · [h(t−1), x(t)] + bg ) cell candidate
 *   o(t) = σ( Wo · [h(t−1), x(t)] + bo )   output gate
 *   c(t) = f(t) ⊙ c(t−1) + i(t) ⊙ g(t)   cell state
 *   h(t) = o(t) ⊙ tanh(c(t))              hidden state
 *
 * **Input layout**: (inputSize × seqLen, batchSize) — same convention as
 * the `Recurrent` layer.
 *
 * **Output**: (hiddenSize, batchSize) — the last hidden state h(T).
 *
 * Weights for all four gates are stored in a single concatenated parameter
 * matrix W of shape (4 × hiddenSize, hiddenSize + inputSize) to allow a
 * single matrix multiply per time step, reducing PHP function-call overhead.
 *
 * Gradients are clipped element-wise to ±`GRAD_CLIP` to prevent explosion.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      The Rubix ML Community
 */
class LSTM implements Hidden, Parametric
{
    /**
     * Element-wise gradient clip threshold.
     */
    protected const GRAD_CLIP = 5.0;

    /**
     * Gate index offsets within the stacked weight matrix.
     */
    protected const GATE_F = 0; // forget
    protected const GATE_I = 1; // input
    protected const GATE_G = 2; // cell candidate (tanh gate)
    protected const GATE_O = 3; // output

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
     * Stacked gate weight matrix W: (4*H, H+I) — [Wf; Wi; Wg; Wo].
     *
     * @var Parameter|null
     */
    protected ?Parameter $w = null;

    /**
     * Stacked gate bias vector b: (4*H,) — [bf; bi; bg; bo].
     *
     * @var Parameter|null
     */
    protected ?Parameter $b = null;

    /**
     * Inferred sequence length.
     *
     * @var positive-int|null
     */
    protected ?int $seqLen = null;

    /**
     * Cached hidden states h(0)…h(T) from the last forward pass.
     *
     * @var array[]|null  Each entry: float[][]
     */
    protected ?array $hStates = null;

    /**
     * Cached cell states c(0)…c(T) from the last forward pass.
     *
     * @var array[]|null
     */
    protected ?array $cStates = null;

    /**
     * Cached gate pre-activations z(t) = W · [h;x] + b at each step.
     * Shape per step: (4H, B).
     *
     * @var array[]|null
     */
    protected ?array $gates = null;

    /**
     * Cached per-step inputs x(t): (I, B).
     *
     * @var array[]|null
     */
    protected ?array $inputs = null;

    /**
     * @param int $hiddenSize Number of hidden units (≥ 1)
     * @param int $inputSize  Features per time step (≥ 1)
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
     * @return positive-int
     */
    public function width() : int
    {
        return $this->hiddenSize;
    }

    /**
     * Initialise stacked weight matrix and bias vector.
     *
     * @param positive-int $fanIn  = inputSize × seqLen
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

        $H    = $this->hiddenSize;
        $I    = $this->inputSize;
        $init = new Xavier2();

        // W: (4H, H+I) — initialise each gate block separately then stack
        $rows = [];

        for ($g = 0; $g < 4; $g++) {
            $block = $init->initialize($H + $I, $H)->asArray();

            foreach ($block as $row) {
                $rows[] = $row;
            }
        }

        $this->w = new Parameter(Matrix::quick($rows));
        $this->b = new Parameter(
            (new Constant(0.0))->initialize(1, 4 * $H)->columnAsVector(0)
        );

        // Initialise forget-gate bias to 1 to encourage remembering early in training
        $bArr = $this->b->param()->asArray();

        for ($h = 0; $h < $H; $h++) {
            $bArr[$h] = 1.0;  // forget-gate bias
        }

        $this->b = new Parameter(Vector::quick($bArr));

        return $H;
    }

    /**
     * Forward pass: compute all time steps, cache internals, return h(T).
     *
     * @param Matrix $input (I*T, B)
     * @throws RuntimeException
     * @return Matrix (H, B)
     */
    public function forward(Matrix $input) : Matrix
    {
        if ($this->w === null || $this->seqLen === null) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        [$hStates, $cStates, $gates, $inputs] = $this->runForward($input, true);

        $this->hStates = $hStates;
        $this->cStates = $cStates;
        $this->gates   = $gates;
        $this->inputs  = $inputs;

        return Matrix::quick($hStates[$this->seqLen]);
    }

    /**
     * Inference pass (no caching).
     *
     * @param Matrix $input
     * @throws RuntimeException
     * @return Matrix (H, B)
     */
    public function infer(Matrix $input) : Matrix
    {
        if ($this->w === null || $this->seqLen === null) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        [$hStates] = $this->runForward($input, false);

        return Matrix::quick($hStates[$this->seqLen]);
    }

    /**
     * Backward pass (BPTT through all time steps).
     *
     * @param Deferred  $prevGradient  dL/dh(T): (H, B)
     * @param Optimizer $optimizer
     * @throws RuntimeException
     * @return Deferred  dL/dInput: (I*T, B)
     */
    public function back(Deferred $prevGradient, Optimizer $optimizer) : Deferred
    {
        if ($this->w === null || $this->seqLen === null) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        if ($this->hStates === null) {
            throw new RuntimeException('Must perform a forward pass'
                . ' before backpropagating.');
        }

        $dHNext = $prevGradient()->asArray(); // (H, B)

        $T      = $this->seqLen;
        $H      = $this->hiddenSize;
        $I      = $this->inputSize;
        $B      = count($dHNext[0]);
        $wArr   = $this->w->param()->asArray();

        $dW     = $this->zeroMatrix(4 * $H, $H + $I);
        $dB     = $this->zeroVector(4 * $H);
        $dInput = $this->zeroMatrix($I * $T, $B);

        $dCNext = $this->zeroMatrix($H, $B);

        for ($t = $T - 1; $t >= 0; $t--) {
            $gArr   = $this->gates[$t];   // (4H, B) — raw gate pre-activations
            $hPrev  = $this->hStates[$t]; // (H, B) — h(t-1)
            $cPrev  = $this->cStates[$t]; // (H, B) — c(t-1)
            $cCurr  = $this->cStates[$t + 1]; // c(t)
            $xtArr  = $this->inputs[$t];  // (I, B) — x(t)

            // Compute gate activations
            $f = $this->sigArr($gArr, 0 * $H, $H, $B);
            $i = $this->sigArr($gArr, 1 * $H, $H, $B);
            $g = $this->tanhArr($gArr, 2 * $H, $H, $B);
            $o = $this->sigArr($gArr, 3 * $H, $H, $B);

            $tanhC = $this->tanhVals($cCurr, $H, $B);

            // dh/dc contribution: dL/dc(t) = dL/dh(t) ⊙ o(t) ⊙ (1 - tanh²(c(t))) + dL/dc(t+1) ⊙ f(t+1)
            // Here dCNext already incorporates future contribution
            $dC = $this->zeroMatrix($H, $B);

            for ($h = 0; $h < $H; $h++) {
                for ($b = 0; $b < $B; $b++) {
                    $dC[$h][$b] = $dHNext[$h][$b] * $o[$h][$b] * (1.0 - $tanhC[$h][$b] ** 2)
                                + $dCNext[$h][$b];
                }
            }

            // Gate deltas
            $dF = $this->zeroMatrix($H, $B);
            $dI = $this->zeroMatrix($H, $B);
            $dG = $this->zeroMatrix($H, $B);
            $dO = $this->zeroMatrix($H, $B);

            for ($h = 0; $h < $H; $h++) {
                for ($b = 0; $b < $B; $b++) {
                    // δf = dC ⊙ c(t−1) ⊙ f ⊙ (1−f)
                    $dF[$h][$b] = $dC[$h][$b] * $cPrev[$h][$b]
                                * $f[$h][$b] * (1.0 - $f[$h][$b]);

                    // δi = dC ⊙ g ⊙ i ⊙ (1−i)
                    $dI[$h][$b] = $dC[$h][$b] * $g[$h][$b]
                                * $i[$h][$b] * (1.0 - $i[$h][$b]);

                    // δg = dC ⊙ i ⊙ (1 − g²)
                    $dG[$h][$b] = $dC[$h][$b] * $i[$h][$b] * (1.0 - $g[$h][$b] ** 2);

                    // δo = dH ⊙ tanh(c) ⊙ o ⊙ (1−o)
                    $dO[$h][$b] = $dHNext[$h][$b] * $tanhC[$h][$b]
                                * $o[$h][$b] * (1.0 - $o[$h][$b]);
                }
            }

            // Stack gate deltas: (4H, B)
            $delta = $this->stackDeltas($dF, $dI, $dG, $dO, $H, $B);

            // Build [h(t-1); x(t)]: ((H+I), B)
            $hx = $this->stackHX($hPrev, $xtArr, $H, $I, $B);

            // dW += delta · hx^T  — (4H, H+I)
            for ($r = 0; $r < 4 * $H; $r++) {
                for ($c = 0; $c < $H + $I; $c++) {
                    $sum = 0.0;

                    for ($b = 0; $b < $B; $b++) {
                        $sum += $delta[$r][$b] * $hx[$c][$b];
                    }

                    $dW[$r][$c] += $sum;
                }
            }

            // dB += Σ_b delta
            for ($r = 0; $r < 4 * $H; $r++) {
                $s = 0.0;

                for ($b = 0; $b < $B; $b++) {
                    $s += $delta[$r][$b];
                }

                $dB[$r] += $s;
            }

            // dH for previous step: W[:, 0:H]^T · delta
            $newDH = $this->zeroMatrix($H, $B);

            for ($h = 0; $h < $H; $h++) {
                for ($b = 0; $b < $B; $b++) {
                    $val = 0.0;

                    for ($r = 0; $r < 4 * $H; $r++) {
                        $val += $wArr[$r][$h] * $delta[$r][$b];
                    }

                    $newDH[$h][$b] = $val;
                }
            }

            // dInput(t) = W[:, H:H+I]^T · delta
            for ($ii = 0; $ii < $I; $ii++) {
                $rowIdx = $t * $I + $ii;

                for ($b = 0; $b < $B; $b++) {
                    $val = 0.0;

                    for ($r = 0; $r < 4 * $H; $r++) {
                        $val += $wArr[$r][$H + $ii] * $delta[$r][$b];
                    }

                    $dInput[$rowIdx][$b] = $val;
                }
            }

            // dC(t-1) = dC ⊙ f(t)
            $newDC = $this->zeroMatrix($H, $B);

            for ($h = 0; $h < $H; $h++) {
                for ($b = 0; $b < $B; $b++) {
                    $newDC[$h][$b] = $dC[$h][$b] * $f[$h][$b];
                }
            }

            $dHNext = $newDH;
            $dCNext = $newDC;
        }

        $dWMat = $this->clipGrad(Matrix::quick($dW));
        $dBVec = $this->clipVec(Vector::quick($dB));

        $this->w->update($dWMat, $optimizer);
        $this->b->update($dBVec, $optimizer);

        $this->hStates = null;
        $this->cStates = null;
        $this->gates   = null;
        $this->inputs  = null;

        $dInputMat = Matrix::quick($dInput);

        return new Deferred(static fn () => $dInputMat);
    }

    /**
     * Yield trainable parameters.
     *
     * @throws RuntimeException
     * @return Generator<Parameter>
     */
    public function parameters() : Generator
    {
        if ($this->w === null) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        yield 'w' => $this->w;
        yield 'b' => $this->b;
    }

    /**
     * Restore parameters from a Snapshot.
     *
     * @param Parameter[] $parameters
     */
    public function restore(array $parameters) : void
    {
        $this->w = $parameters['w'];
        $this->b = $parameters['b'];
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'LSTM ('
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
     * Run the full forward pass through all time steps.
     *
     * @param Matrix $input  (I*T, B)
     * @param bool   $cache
     * @return array{array[], array[], array[], array[]}
     */
    protected function runForward(Matrix $input, bool $cache) : array
    {
        $T       = $this->seqLen;
        $H       = $this->hiddenSize;
        $I       = $this->inputSize;
        $B       = $input->n();
        $inpArr  = $input->asArray();
        $wArr    = $this->w->param()->asArray();
        $bArr    = $this->b->param()->asArray();

        $hArr    = $this->zeroMatrix($H, $B);
        $cArr    = $this->zeroMatrix($H, $B);

        $hStates = [$hArr];
        $cStates = [$cArr];
        $gates   = [];
        $inputs  = [];

        for ($t = 0; $t < $T; $t++) {
            // x(t): (I, B)
            $xt = $this->zeroMatrix($I, $B);

            for ($ii = 0; $ii < $I; $ii++) {
                $xt[$ii] = $inpArr[$t * $I + $ii];
            }

            // z = W · [h; x] + b — (4H, B)
            $HI = $H + $I;
            $z  = $this->zeroMatrix(4 * $H, $B);

            for ($r = 0; $r < 4 * $H; $r++) {
                $bias = $bArr[$r];

                for ($b = 0; $b < $B; $b++) {
                    $val = $bias;

                    for ($hh = 0; $hh < $H; $hh++) {
                        $val += $wArr[$r][$hh] * $hArr[$hh][$b];
                    }

                    for ($ii = 0; $ii < $I; $ii++) {
                        $val += $wArr[$r][$H + $ii] * $xt[$ii][$b];
                    }

                    $z[$r][$b] = $val;
                }
            }

            // Gate activations
            $f = $this->sigArr($z, 0 * $H, $H, $B);
            $i = $this->sigArr($z, 1 * $H, $H, $B);
            $g = $this->tanhArr($z, 2 * $H, $H, $B);
            $o = $this->sigArr($z, 3 * $H, $H, $B);

            // c(t) = f ⊙ c(t-1) + i ⊙ g
            $cNew = $this->zeroMatrix($H, $B);

            for ($h = 0; $h < $H; $h++) {
                for ($b = 0; $b < $B; $b++) {
                    $cNew[$h][$b] = $f[$h][$b] * $cArr[$h][$b]
                                  + $i[$h][$b] * $g[$h][$b];
                }
            }

            // h(t) = o ⊙ tanh(c)
            $hNew = $this->zeroMatrix($H, $B);

            for ($h = 0; $h < $H; $h++) {
                for ($b = 0; $b < $B; $b++) {
                    $hNew[$h][$b] = $o[$h][$b] * tanh($cNew[$h][$b]);
                }
            }

            $hArr = $hNew;
            $cArr = $cNew;

            $hStates[] = $hNew;
            $cStates[] = $cNew;

            if ($cache) {
                $gates[]  = $z;
                $inputs[] = $xt;
            }
        }

        return [$hStates, $cStates, $gates, $inputs];
    }

    /**
     * Apply sigmoid to a slice of a 2-D gate array: rows [offset, offset+H).
     *
     * @param float[][] $z
     * @return float[][]
     */
    protected function sigArr(array $z, int $offset, int $H, int $B) : array
    {
        $out = $this->zeroMatrix($H, $B);

        for ($h = 0; $h < $H; $h++) {
            for ($b = 0; $b < $B; $b++) {
                $out[$h][$b] = 1.0 / (1.0 + exp(-$z[$offset + $h][$b]));
            }
        }

        return $out;
    }

    /**
     * Apply tanh to a slice of a 2-D gate array.
     *
     * @param float[][] $z
     * @return float[][]
     */
    protected function tanhArr(array $z, int $offset, int $H, int $B) : array
    {
        $out = $this->zeroMatrix($H, $B);

        for ($h = 0; $h < $H; $h++) {
            for ($b = 0; $b < $B; $b++) {
                $out[$h][$b] = tanh($z[$offset + $h][$b]);
            }
        }

        return $out;
    }

    /**
     * Apply tanh element-wise to a 2-D array (for tanh(c)).
     *
     * @param float[][] $c
     * @return float[][]
     */
    protected function tanhVals(array $c, int $H, int $B) : array
    {
        $out = $this->zeroMatrix($H, $B);

        for ($h = 0; $h < $H; $h++) {
            for ($b = 0; $b < $B; $b++) {
                $out[$h][$b] = tanh($c[$h][$b]);
            }
        }

        return $out;
    }

    /**
     * Stack four gate delta arrays into (4H, B).
     *
     * @param float[][] $dF
     * @param float[][] $dI
     * @param float[][] $dG
     * @param float[][] $dO
     * @return float[][]
     */
    protected function stackDeltas(array $dF, array $dI, array $dG, array $dO, int $H, int $B) : array
    {
        $out = [];

        for ($h = 0; $h < $H; $h++) {
            $out[]          = $dF[$h];
        }

        for ($h = 0; $h < $H; $h++) {
            $out[]          = $dI[$h];
        }

        for ($h = 0; $h < $H; $h++) {
            $out[]          = $dG[$h];
        }

        for ($h = 0; $h < $H; $h++) {
            $out[]          = $dO[$h];
        }

        return $out;
    }

    /**
     * Stack [h(t-1); x(t)] into ((H+I), B).
     *
     * @param float[][] $h  (H, B)
     * @param float[][] $x  (I, B)
     * @return float[][]  (H+I, B)
     */
    protected function stackHX(array $h, array $x, int $H, int $I, int $B) : array
    {
        return array_merge($h, $x);
    }

    /**
     * @return float[][]
     */
    protected function zeroMatrix(int $rows, int $cols) : array
    {
        $row = array_fill(0, $cols, 0.0);

        return array_fill(0, $rows, $row);
    }

    /**
     * @return float[]
     */
    protected function zeroVector(int $n) : array
    {
        return array_fill(0, $n, 0.0);
    }

    /**
     * Clip a Matrix gradient element-wise.
     */
    protected function clipGrad(Matrix $g) : Matrix
    {
        $c = self::GRAD_CLIP;

        return $g->map(fn ($v) => max(-$c, min($c, $v)));
    }

    /**
     * Clip a Vector gradient element-wise.
     */
    protected function clipVec(Vector $v) : Vector
    {
        $c = self::GRAD_CLIP;

        return $v->map(fn ($x) => max(-$c, min($c, $x)));
    }
}
