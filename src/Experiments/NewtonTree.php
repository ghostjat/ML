<?php

declare(strict_types=1);

namespace Rubix\ML\Experiments;

use function count;
use function max;
use function min;
use function mt_rand;
use function range;

class NewtonTree
{
    private int $maxDepth;
    private float $minChildWeight;
    private float $lambda;
    private float $gamma;
    private float $colsampleBytree;
    private float $maxDeltaStep;
    private int $numBins;

    private array $columns = [];
    private array $g = [];
    private array $h = [];

    public ?NewtonNode $root = null;

    public function __construct(
        int $maxDepth = 6,
        float $minChildWeight = 1.0,
        float $lambda = 1.0,
        float $gamma = 0.0,
        float $colsampleBytree = 1.0,
        float $maxDeltaStep = 0.0,
        int $numBins = 256 // Bins 0-255 for data, 256 for missing
    ) {
        $this->maxDepth = $maxDepth;
        $this->minChildWeight = $minChildWeight;
        $this->lambda = $lambda;
        $this->gamma = $gamma;
        $this->colsampleBytree = $colsampleBytree;
        $this->maxDeltaStep = $maxDeltaStep;
        $this->numBins = $numBins;
    }

    public function train(array &$columns, array &$g, array &$h, array $indices): void
    {
        $this->columns = &$columns;
        $this->g = &$g;
        $this->h = &$h;

        $this->root = $this->build($indices, 0);
    }

    private function build(array $indices, int $depth): NewtonNode
    {
        $node = new NewtonNode();

        $G = 0.0;
        $H = 0.0;

        foreach ($indices as $i) {
            $G += $this->g[$i];
            $H += $this->h[$i];
        }

        // XGBoost Regularized Leaf Weight (with dynamic max_delta_step clamping)
        $denominator = $H + $this->lambda + 1e-8;
        $rawLeaf = -$G / $denominator;
        if ($this->maxDeltaStep > 0.0) {
            $node->leafValue = max(-$this->maxDeltaStep, min($this->maxDeltaStep, $rawLeaf));
        } else {
            $node->leafValue = $rawLeaf;
        }

        if ($depth >= $this->maxDepth || $H < $this->minChildWeight || count($indices) < 4) {
            return $node;
        }

        $bestGain = 0.0;
        $bestFeature = -1;
        $bestSplitBin = -1;
        $bestDefaultLeft = false;

        $featureCount = count($this->columns);
        $numCols = (int) round($featureCount * $this->colsampleBytree);
        $numCols = max(1, min($numCols, $featureCount));
        
        $featureOrder = range(0, $featureCount - 1);
        if ($numCols < $featureCount) {
            // Fast Fisher-Yates shuffle with mt_rand for speed
            for ($j = 0; $j < $numCols; $j++) {
                $r = mt_rand($j, $featureCount - 1);
                $tmp = $featureOrder[$j];
                $featureOrder[$j] = $featureOrder[$r];
                $featureOrder[$r] = $tmp;
            }
        }

        // Pre-allocate histogram buffers ONCE per node
        $histSize = $this->numBins + 1; 
        $histG = array_fill(0, $histSize, 0.0);
        $histH = array_fill(0, $histSize, 0.0);

        for ($fIdx = 0; $fIdx < $numCols; $fIdx++) {
            $f = $featureOrder[$fIdx];
            $col = &$this->columns[$f];

            // Zero out histograms without triggering GC array reallocation
            for ($b = 0; $b < $histSize; $b++) {
                $histG[$b] = 0.0;
                $histH[$b] = 0.0;
            }

            // O(N) Histogram Construction (No sorting!)
            foreach ($indices as $i) {
                $bin = $col[$i]; 
                $histG[$bin] += $this->g[$i];
                $histH[$bin] += $this->h[$i];
            }

            $missingG = $histG[$this->numBins];
            $missingH = $histH[$this->numBins];

            $GL_right = 0.0;
            $HL_right = 0.0;

            // One-Pass Sparsity Optimization
            for ($bin = 0; $bin < $this->numBins - 1; $bin++) {
                $GL_right += $histG[$bin];
                $HL_right += $histH[$bin];

                // 1. Evaluate Default Right
                $GR_right = $G - $GL_right - $missingG;
                $HR_right = $H - $HL_right - $missingH;

                if ($HL_right >= $this->minChildWeight && ($HR_right + $missingH) >= $this->minChildWeight) {
                    $gainRight = $this->calculateGain($G, $H, $GL_right, $HL_right, $GR_right + $missingG, $HR_right + $missingH);
                    if ($gainRight > $bestGain) {
                        $bestGain = $gainRight;
                        $bestFeature = $f;
                        $bestSplitBin = $bin;
                        $bestDefaultLeft = false;
                    }
                }

                // 2. Evaluate Default Left (Add missing directly to the left side mathematically)
                $GL_left = $GL_right + $missingG;
                $HL_left = $HL_right + $missingH;
                
                $GR_left = $G - $GL_left;
                $HR_left = $H - $HL_left;

                if ($HL_left >= $this->minChildWeight && $HR_left >= $this->minChildWeight) {
                    $gainLeft = $this->calculateGain($G, $H, $GL_left, $HL_left, $GR_left, $HR_left);
                    if ($gainLeft > $bestGain) {
                        $bestGain = $gainLeft;
                        $bestFeature = $f;
                        $bestSplitBin = $bin;
                        $bestDefaultLeft = true;
                    }
                }
            }
        }

        if ($bestFeature === -1) {
            return $node;
        }

        $node->isLeaf = false;
        $node->splitFeature = $bestFeature;
        $node->splitBin = $bestSplitBin;
        $node->defaultLeft = $bestDefaultLeft;

        $left = [];
        $right = [];
        $col = &$this->columns[$bestFeature];

        foreach ($indices as $i) {
            $val = $col[$i];
            if ($val === $this->numBins) { // Missing value routes based on best default
                if ($bestDefaultLeft) {
                    $left[] = $i;
                } else {
                    $right[] = $i;
                }
            } elseif ($val <= $bestSplitBin) {
                $left[] = $i;
            } else {
                $right[] = $i;
            }
        }

        $node->left = $this->build($left, $depth + 1);
        $node->right = $this->build($right, $depth + 1);

        return $node;
    }

    private function calculateGain(float $G, float $H, float $GL, float $HL, float $GR, float $HR): float
    {
        return 
            ($GL * $GL) / ($HL + $this->lambda) +
            ($GR * $GR) / ($HR + $this->lambda) -
            ($G * $G) / ($H + $this->lambda) -
            $this->gamma;
    }

    public function predictSample(array $sample): float
    {
        $node = $this->root;

        while (!$node->isLeaf) {
            $val = $sample[$node->splitFeature];
            
            if ($val === $this->numBins) {
                $node = $node->defaultLeft ? $node->left : $node->right;
            } else {
                $node = $val <= $node->splitBin ? $node->left : $node->right;
            }
        }

        return $node->leafValue;
    }
}

class NewtonNode
{
    public bool $isLeaf = true;
    public float $leafValue = 0.0;
    public int $splitFeature = 0;
    public int $splitBin = 0;
    public bool $defaultLeft = false;
    public ?NewtonNode $left = null;
    public ?NewtonNode $right = null;
}