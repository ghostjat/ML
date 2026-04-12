<?php

declare(strict_types=1);

namespace Rubix\ML\NeuralNet\Layers;

use Tensor\Matrix;
use Tensor\Vector;
use Rubix\ML\Deferred;
use Rubix\ML\Helpers\Params;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Initializers\He;
use Rubix\ML\NeuralNet\Initializers\Constant;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Generator;

use function count;
use function array_fill;

/**
 * Conv1D
 *
 * A 1-D convolutional hidden layer for sequence data. It applies a bank of
 * learnable filters (kernels) of a fixed width to every position along the
 * time axis, producing a new sequence of activation maps.
 *
 * **Input layout** (as expected from the previous layer):
 *   Matrix of shape (inputChannels × sequenceLength, batchSize)
 *   i.e. rows are interleaved as: channel0_step0, channel0_step1, …,
 *        channel1_step0, …
 *
 * **Output layout**:
 *   Matrix of shape (numFilters × outputLength, batchSize)
 *   where outputLength = ⌊(sequenceLength − kernelSize) / stride⌋ + 1
 *
 * The layer implements the standard im2col-free direct convolution suitable
 * for small-to-medium PHP workloads. All weight gradients are computed via
 * the chain rule, matching the backpropagation contract of the Hidden interface.
 *
 * Place Conv1D inside a `FeedForward` network after a `Placeholder1D` input
 * layer whose width equals (inputChannels × sequenceLength). Typically follow
 * Conv1D with an `Activation` layer and one or more `Dense` layers.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      The Rubix ML Community
 */
class Conv1D implements Hidden, Parametric
{
    /**
     * The number of output filters (channels).
     *
     * @var positive-int
     */
    protected int $numFilters;

    /**
     * The width of each convolutional kernel.
     *
     * @var positive-int
     */
    protected int $kernelSize;

    /**
     * The number of input channels per time step.
     *
     * @var positive-int
     */
    protected int $inputChannels;

    /**
     * The stride of the convolution.
     *
     * @var positive-int
     */
    protected int $stride;

    /**
     * Weight initializer for the kernels.
     *
     * @var Initializer
     */
    protected Initializer $initializer;

    /**
     * Kernel parameter: shape (numFilters × inputChannels * kernelSize).
     *
     * @var Parameter|null
     */
    protected ?Parameter $kernels = null;

    /**
     * Bias parameter: shape (numFilters,).
     *
     * @var Parameter|null
     */
    protected ?Parameter $biases = null;

    /**
     * Inferred sequence length (set during initialize).
     *
     * @var positive-int|null
     */
    protected ?int $seqLen = null;

    /**
     * Cached input from the last forward pass (for backprop).
     *
     * @var Matrix|null
     */
    protected ?Matrix $input = null;

    /**
     * @param int              $numFilters    Number of output filters
     * @param int              $kernelSize    Width of each kernel (≥ 1)
     * @param int              $inputChannels Input channels per time step (≥ 1)
     * @param int              $stride        Convolution stride (≥ 1)
     * @param Initializer|null $initializer   Kernel weight initializer
     * @throws InvalidArgumentException
     */
    public function __construct(
        int $numFilters = 32,
        int $kernelSize = 3,
        int $inputChannels = 1,
        int $stride = 1,
        ?Initializer $initializer = null
    ) {
        if ($numFilters < 1) {
            throw new InvalidArgumentException("Number of filters must be"
                . " at least 1, $numFilters given.");
        }

        if ($kernelSize < 1) {
            throw new InvalidArgumentException("Kernel size must be"
                . " at least 1, $kernelSize given.");
        }

        if ($inputChannels < 1) {
            throw new InvalidArgumentException("Input channels must be"
                . " at least 1, $inputChannels given.");
        }

        if ($stride < 1) {
            throw new InvalidArgumentException("Stride must be"
                . " at least 1, $stride given.");
        }

        $this->numFilters    = $numFilters;
        $this->kernelSize    = $kernelSize;
        $this->inputChannels = $inputChannels;
        $this->stride        = $stride;
        $this->initializer   = $initializer ?? new He();
    }

    /**
     * Return the output width of this layer.
     *
     * @throws RuntimeException
     * @return positive-int
     */
    public function width() : int
    {
        if ($this->seqLen === null) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        return $this->numFilters * $this->outputLength();
    }

    /**
     * Initialize kernel and bias parameters.
     *
     * fanIn must equal inputChannels × sequenceLength.
     *
     * @param positive-int $fanIn
     * @throws InvalidArgumentException
     * @return positive-int fanOut = numFilters × outputLength
     */
    public function initialize(int $fanIn) : int
    {
        if ($fanIn % $this->inputChannels !== 0) {
            throw new InvalidArgumentException(
                "fanIn ($fanIn) must be divisible by inputChannels"
                . " ({$this->inputChannels})."
            );
        }

        $this->seqLen = $fanIn / $this->inputChannels;

        if ($this->seqLen < $this->kernelSize) {
            throw new InvalidArgumentException(
                "Sequence length ({$this->seqLen}) must be at least"
                . " kernelSize ({$this->kernelSize})."
            );
        }

        $kernelFanIn  = $this->inputChannels * $this->kernelSize;
        $kernelMatrix = $this->initializer->initialize($kernelFanIn, $this->numFilters);

        $this->kernels = new Parameter($kernelMatrix);

        $biasMatrix  = (new Constant(0.0))->initialize(1, $this->numFilters);
        $this->biases = new Parameter($biasMatrix->columnAsVector(0));

        return $this->numFilters * $this->outputLength();
    }

    /**
     * Compute the forward pass.
     *
     * @param Matrix $input Shape: (inputChannels * seqLen, batchSize)
     * @throws RuntimeException
     * @return Matrix Shape: (numFilters * outputLength, batchSize)
     */
    public function forward(Matrix $input) : Matrix
    {
        if ($this->kernels === null || $this->seqLen === null) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        $this->input = $input;

        return $this->convolve($input);
    }

    /**
     * Compute the inference pass (identical to forward for Conv1D).
     *
     * @param Matrix $input
     * @throws RuntimeException
     * @return Matrix
     */
    public function infer(Matrix $input) : Matrix
    {
        if ($this->kernels === null || $this->seqLen === null) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        return $this->convolve($input);
    }

    /**
     * Compute the backward pass: update kernel/bias parameters and
     * return the gradient w.r.t. the layer input.
     *
     * @param Deferred $prevGradient Gradient w.r.t. this layer's output
     * @param Optimizer $optimizer
     * @throws RuntimeException
     * @return Deferred Gradient w.r.t. this layer's input
     */
    public function back(Deferred $prevGradient, Optimizer $optimizer) : Deferred
    {
        if ($this->kernels === null || $this->seqLen === null) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        if ($this->input === null) {
            throw new RuntimeException('Must perform a forward pass'
                . ' before backpropagating.');
        }

        $dOut   = $prevGradient();   // (numFilters * outLen, batchSize)
        $input  = $this->input;
        $this->input = null;

        $kernels = $this->kernels->param();

        // Compute kernel and bias gradients, then update
        [$dKernels, $dBiases, $dInput] = $this->computeGradients($input, $dOut);

        $this->kernels->update($dKernels, $optimizer);
        $this->biases->update($dBiases, $optimizer);

        return new Deferred(static fn () => $dInput);
    }

    /**
     * Yield the trainable parameters of this layer.
     *
     * @throws RuntimeException
     * @return Generator<Parameter>
     */
    public function parameters() : Generator
    {
        if ($this->kernels === null) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        yield 'kernels' => $this->kernels;
        yield 'biases'  => $this->biases;
    }

    /**
     * Restore parameters (e.g. from a Snapshot).
     *
     * @param Parameter[] $parameters
     */
    public function restore(array $parameters) : void
    {
        $this->kernels = $parameters['kernels'];
        $this->biases  = $parameters['biases'];
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Conv1D ('
            . Params::stringify([
                'filters'       => $this->numFilters,
                'kernel size'   => $this->kernelSize,
                'input channels'=> $this->inputChannels,
                'stride'        => $this->stride,
            ])
            . ')';
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /**
     * Compute the output length given the current sequence length and stride.
     *
     * @return positive-int
     */
    protected function outputLength() : int
    {
        return (int) (($this->seqLen - $this->kernelSize) / $this->stride) + 1;
    }

    /**
     * Direct 1-D convolution: input (C_in * L, B) → output (F * L_out, B).
     *
     * @param Matrix $input
     * @return Matrix
     */
    protected function convolve(Matrix $input) : Matrix
    {
        $kernelArr = $this->kernels->param()->asArray();
        $biasArr   = $this->biases->param()->asArray();
        $inputArr  = $input->asArray();
        $batchSize = $input->n();
        $outLen    = $this->outputLength();
        $C         = $this->inputChannels;
        $L         = $this->seqLen;
        $K         = $this->kernelSize;
        $F         = $this->numFilters;
        $S         = $this->stride;

        $output = [];

        for ($f = 0; $f < $F; $f++) {
            $bias = $biasArr[$f];

            for ($t = 0; $t < $outLen; $t++) {
                $tStart = $t * $S;
                $row    = [];

                for ($b = 0; $b < $batchSize; $b++) {
                    $val = $bias;

                    for ($c = 0; $c < $C; $c++) {
                        for ($k = 0; $k < $K; $k++) {
                            $inputRow    = $c * $L + ($tStart + $k);
                            $kernelCol   = $c * $K + $k;
                            $val += $kernelArr[$f][$kernelCol] * $inputArr[$inputRow][$b];
                        }
                    }

                    $row[] = $val;
                }

                $output[$f * $outLen + $t] = $row;
            }
        }

        return Matrix::quick($output);
    }

    /**
     * Compute gradients for kernels, biases, and the input.
     *
     * dL/dKernel[f, c, k] = Σ_{t,b} dOut[f*outLen+t, b] · input[c*L+(tStart+k), b]
     * dL/dBias[f]          = Σ_{t,b} dOut[f*outLen+t, b]
     * dL/dInput[c*L+i, b]  = Σ_{f,k} kernel[f,c,k] · dOut[f*outLen+((i−k)/S), b]
     *
     * @param Matrix $input   (C*L, B)
     * @param Matrix $dOut    (F*outLen, B)
     * @return array{Matrix, Vector, Matrix}
     */
    protected function computeGradients(Matrix $input, Matrix $dOut) : array
    {
        $inputArr  = $input->asArray();
        $dOutArr   = $dOut->asArray();
        $kernelArr = $this->kernels->param()->asArray();
        $batchSize = $input->n();
        $outLen    = $this->outputLength();
        $C         = $this->inputChannels;
        $L         = $this->seqLen;
        $K         = $this->kernelSize;
        $F         = $this->numFilters;
        $S         = $this->stride;

        // Allocate gradient accumulators
        $dKernels = array_fill(0, $F, array_fill(0, $C * $K, 0.0));
        $dBiases  = array_fill(0, $F, 0.0);
        $dInput   = array_fill(0, $C * $L, array_fill(0, $batchSize, 0.0));

        for ($f = 0; $f < $F; $f++) {
            for ($t = 0; $t < $outLen; $t++) {
                $tStart   = $t * $S;
                $dOutRow  = $dOutArr[$f * $outLen + $t];

                // Bias gradient
                foreach ($dOutRow as $g) {
                    $dBiases[$f] += $g;
                }

                for ($c = 0; $c < $C; $c++) {
                    for ($k = 0; $k < $K; $k++) {
                        $inputRow  = $c * $L + ($tStart + $k);
                        $kernelCol = $c * $K + $k;

                        // Kernel gradient: Σ_b dOut[f*outLen+t, b] * input[inputRow, b]
                        $dkSum = 0.0;

                        for ($b = 0; $b < $batchSize; $b++) {
                            $dkSum += $dOutRow[$b] * $inputArr[$inputRow][$b];
                        }

                        $dKernels[$f][$kernelCol] += $dkSum;

                        // Input gradient: accumulate kernel[f,c,k] * dOut[f*outLen+t, b]
                        $kv = $kernelArr[$f][$kernelCol];

                        for ($b = 0; $b < $batchSize; $b++) {
                            $dInput[$inputRow][$b] += $kv * $dOutRow[$b];
                        }
                    }
                }
            }
        }

        return [
            Matrix::quick($dKernels),
            Vector::quick($dBiases),
            Matrix::quick($dInput),
        ];
    }
}
