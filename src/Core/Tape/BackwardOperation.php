<?php

namespace NumPower\Tensor\Core\Tape;

use NDArray as nd;
use NumPower\Tensor\Variable;

class BackwardOperation
{
    public static function add(Variable $output, bool $benchmark,  \NDArray $grad, Variable $a, Variable $b): void
    {
        $a->backward($grad, benchmark: $benchmark);
        $b->backward($grad, benchmark: $benchmark);
    }

    public static function subtract(Variable $output, bool $benchmark,  \NDArray $grad, Variable $a, Variable $b): void
    {
        $a->backward($grad, benchmark: $benchmark);
        $b->backward(-$grad, benchmark: $benchmark);
    }

    public static function multiply(Variable $output, bool $benchmark,  \NDArray $grad, Variable $a, Variable $b): void
    {
        $a->backward($grad * $b->getArray(), benchmark: $benchmark);
        $b->backward($a->getArray() * $grad, benchmark: $benchmark);
    }

    public static function matmul(Variable $output, bool $benchmark,  \NDArray $grad, Variable $a, Variable $b): void
    {
        $start = microtime(true);
        $b_transpose = nd::transpose($b->getArray());
        $a_transpose = nd::transpose($a->getArray());
        $stop = microtime(true);
        if ($benchmark) {
            echo "\n transpose: ". ($stop - $start);
        }

        $start = microtime(true);
        $dx_da = nd::matmul($grad, $b_transpose);
        $dx_db = nd::matmul($a_transpose, $grad);
        $stop = microtime(true);
        if ($benchmark) {
            echo "\n matmul: ". ($stop - $start);
        }
        $a->backward($dx_da, benchmark: $benchmark);
        $b->backward($dx_db, benchmark: $benchmark);
    }

    public static function clip(Variable $output, bool $benchmark,  \NDArray $grad, Variable $a, Variable $min, Variable $max): void
    {
        $greater = nd::greater_equal($a->getArray(), nd::ones($a->getArray()->shape()) * $min->getArray());
        $less = nd::less_equal($a->getArray(), nd::ones($a->getArray()->shape()) * $max->getArray());
        $a->backward($grad * $greater * $less);
    }

    public static function conv2d(Variable $output, bool $benchmark,  \NDArray $grad, Variable $input, Variable $filters, array $strides): void
    {
        [$dW, $dInput] = nd::dnn_conv2d_backward($input->getArray(), $grad, $filters->getArray());
        $input->backward($dInput);
        $filters->backward($dW);
    }

    public static function matrix_rank(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $x): void
    {
        $x->backward(nd::zeros($x->getShape()));
    }

    public static function cond(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $x): void
    {
        $x->backward(nd::zeros($x->getShape()));
    }

    public static function softmax(Variable $output, bool $benchmark,  \NDArray $grad, Variable $x): void
    {
        $new_grad_output = new Variable($output->grad() * $output->getArray());
        $new_grad_output_sum = $new_grad_output->sum_axis(1, True);
        $x->backward($new_grad_output->getArray() - $output->getArray() * $new_grad_output_sum->getArray());
    }

    public static function divide(Variable $output, bool $benchmark,  \NDArray $grad, Variable $a, Variable $b): void
    {
        $a->backward($grad / $b->getArray(), benchmark: $benchmark);
        $b->backward(-$grad * $a->getArray() / $b->getArray() ** 2, benchmark: $benchmark);
    }

    public static function exp(Variable $output, bool $benchmark,  \NDArray $grad, Variable $a): void
    {
        $a->backward($grad * nd::exp($a->getArray()), benchmark: $benchmark);
    }

    public static function exp2(Variable $output, bool $benchmark,  \NDArray $grad, Variable $a): void
    {
        $a->backward($grad * $output->getArray() * M_LN2, benchmark: $benchmark);
    }

    public static function expm1(Variable $output, bool $benchmark,  \NDArray $grad, Variable $a): void
    {
        $a->backward($grad * ($output->getArray() + 1));
    }

    public static function mod(Variable $output, bool $benchmark,  \NDArray $grad, Variable $x, Variable $y): void
    {
        $x->backward($grad);
        $y->backward(nd::zeros($grad->shape()));
    }


    public static function trunc(Variable $output, bool $benchmark,  \NDArray $grad, Variable $a): void
    {
        $a->backward(nd::zeros($grad->shape()));
    }

    public static function floor(Variable $output, bool $benchmark,  \NDArray $grad, Variable $a): void
    {
        $a->backward(nd::zeros($grad->shape()));
    }


    public static function sin(Variable $output, bool $benchmark,  \NDArray $grad, Variable $a): void
    {
        $a->backward($grad * nd::cos($a->getArray()));
    }

    public static function sinh(Variable $output, bool $benchmark,  \NDArray $grad, Variable $a): void
    {
        $a->backward($grad * nd::cosh($a->getArray()));
    }

    public static function ceil(Variable $output, bool $benchmark,  \NDArray $grad, Variable $a): void
    {
        $a->backward(nd::zeros($grad->shape()));
    }

    public static function sinc(Variable $output, bool $benchmark,  \NDArray $grad, Variable $a): void
    {
        $ppi = $a->getArray() * M_PI;
        $squaredPi = $a->getArray() * $a->getArray() * M_PI;
        $a->backward($grad * (($ppi * nd::cos($ppi) - nd::sin($ppi)) / ($squaredPi)));
    }

    public static function mean(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $a): void
    {
        $prod = nd::prod(nd::array($a->getArray()->shape()));
        $out = $grad * nd::ones($a->getArray()->shape()) / $prod;
        if ($a->getArray()->isGPU()) {
            $out = $out->gpu();
        }
        $a->backward($out, benchmark: $benchmark);
    }

    public static function abs(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $a): void
    {
        $a->backward($grad * nd::sign($a->getArray()));
    }

    public static function acos(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $a): void
    {
        $a->backward($grad * -(nd::rsqrt(-$a->getArray() * $a->getArray() + 1)));
    }

    public static function cosh(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $a): void
    {
        $a->backward($grad * nd::sinh($a->getArray()));
    }

    public static function tan(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $a): void
    {
        $a->backward($grad * (1 + ($output->getArray() ** 2)));
    }

    public static function radians(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $a): void
    {
        $a->backward($grad * 0.01745329251994329576923690768488612713);
    }

    public static function arccosh(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $a): void
    {
        $a->backward($grad * nd::rsqrt(($a->getArray() ** 2) - 1));
    }

    public static function arctan(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $a): void
    {
        $a->backward($grad / ($a->getArray() * $a->getArray() + 1));
    }

    public static function arcsin(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $a): void
    {
        $a->backward($grad * nd::rsqrt(-$a->getArray() * $a->getArray() + 1));
    }

    public static function arctanh(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $a): void
    {
        $a->backward($grad * 1 / (1 - ($a->getArray() ** 2)));
    }

    public static function arcsinh(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $a): void
    {
        $a->backward($grad * nd::rsqrt(($a->getArray() ** 2) + 1));
    }

    public static function cos(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $a): void
    {
        $a->backward($grad * -nd::sin($a->getArray()));
    }

    public static function rsqrt(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $a): void
    {
        $a->backward(-0.5 * $grad * ($output->getArray() ** 3));
    }

    public static function reshape(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $a, array $shape): void
    {
        $a->backward(nd::reshape($grad, $a->getArray()->shape()));
    }

    public static function log(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $a): void
    {
        $a->backward($grad / $a->getArray());
    }

    public static function log1p(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $a): void
    {
        $a->backward($grad / ($a->getArray() + 1));
    }

    public static function norm(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $a): void
    {
        $a->backward($a->getArray() / $output->getArray());
    }

    public static function log2(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $a): void
    {
        $a->backward($grad / ($a->getArray() * 0.6931471805599453));
    }

    public static function negative(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $a): void
    {
        $a->backward(-$grad);
    }

    public static function det(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $a): void
    {
        $a->backward(nd::transpose(nd::inv($a->getArray())) * $output->getArray());
    }

    public static function log10(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $a): void
    {
        $a->backward($grad / ($a->getArray() * 2.3025850929940456));
    }

    public static function outer(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $x, Variable $y): void
    {
        $x->backward(nd::sum($y->getArray()) * nd::ones($y->getShape()));
        $y->backward(nd::sum($x->getArray()) * nd::ones($x->getShape()));
    }

    public static function binary_cross_entropy(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $x, Variable $y, float $epsilon): void
    {
        $result = $grad * ($x->getArray() - $y->getArray()) / nd::clip($x->getArray() * (1 - $x->getArray()), $epsilon, PHP_FLOAT_MAX);
        $x->backward($result);
    }

    public static function sqrt(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $a): void
    {
        $a->backward($grad / (2 * $output->getArray()));
    }

    public static function tanh(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $a): void
    {
        $a->backward($grad * (1 - $output->getArray() ** 2));
    }

    public static function sum(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $a, bool $keepDim): void
    {
        $a->backward(nd::ones($a->getArray()->shape()) * $grad);
    }

    public static function sum_axis(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $a, int $axis, bool $keepDim): void
    {
        $ones = nd::ones($a->getArray()->shape());
        if ($a->getArray()->isGPU()) {
            $ones = $ones->gpu();
        }
        $a->backward($ones * $grad);
    }

    public static function cce(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $true, Variable $pred, float $epsilon): void
    {
        $pred_batch = nd::clip($pred->getArray(), $epsilon, 1 - $epsilon);
        $dL_dy_pred = -($true->getArray() / $pred_batch);
        $dL_dy_true = - nd::log($pred->getArray());

        $dL_dy_pred = $dL_dy_pred / count($true->getShape());
        $pred->backward($grad * $dL_dy_true);
        $true->backward($grad * $dL_dy_pred);
    }

    public static function relu(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $a): void
    {
        $a->backward($grad * (nd::greater($a->getArray(), 0)));
    }

    public static function selu(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $a, float $alpha, float $scale): void
    {
        $non_zero = nd::greater($a->getArray(), 0);
        $zeros = nd::less_equal($a->getArray(), 0);
        $zeros = $zeros * ($alpha * (nd::exp($a->getArray())));

        $a->backward($grad * ($scale * ($non_zero + $zeros)));
    }

    public static function celu(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $a, float $alpha): void
    {
        $scale = 1.0;
        $negcoef = $alpha * $scale;
        $poscoef = $scale;
        $negiptcoef = $scale / $alpha;

        $zeros = nd::less_equal($a->getArray(), 0);
        $greater = nd::greater($a->getArray(), 0);
        $cond1 = $zeros * ($output->grad() * $negiptcoef * $negcoef * nd::exp($a->getArray()) * $negiptcoef);
        $cond2 = $greater * ($output->grad() * $poscoef);
        $a->backward($cond1 + $cond2);
    }

    public static function power(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $a, Variable $b): void
    {
        $a->backward($grad * $b->getArray() * $a->getArray() ** ($b->getArray() - 1), benchmark: $benchmark);
        $b->backward($grad * $a->getArray() ** $b->getArray() * nd::log($a->getArray()), benchmark: $benchmark);
    }

    public static function dot(Variable $output, bool $benchmark,  \NDArray|float $grad, Variable $a, Variable $b): void
    {
        $a->backward($grad * $b->getArray());
        $b->backward($grad * $a->getArray());
    }
}