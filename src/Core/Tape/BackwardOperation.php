<?php

namespace NumPower\Tensor\Core\Tape;

use NDArray as nd;
use NumPower\Tensor\Variable;

/**
 * Core backward functions.
 *
 * Implements a static function named after all core operations to calculate
 * partial derivatives.
 */
class BackwardOperation
{
    public static function offsetGet(Variable $output, \NDArray|float $grad, Variable $self, int $offset): void
    {
        $zeros = nd::zeros($self->getShape());
        if (is_float($grad) || count($grad->shape()) < count($zeros->shape())) {
            $zeros[$offset] = $grad;
        } else if (count($grad->shape()) > count($zeros->shape())) {
            $zeros[$offset] = nd::sum($grad);
        }

        $self->diff($zeros);
    }
    public static function add(Variable $output, \NDArray|float $grad, Variable $a, Variable $b): void
    {
        $a->diff($grad);
        $b->diff($grad);
    }
    public static function subtract(Variable $output, \NDArray|float $grad, Variable $a, Variable $b): void
    {
        $a->diff($grad);
        $b->diff(-$grad);
    }
    public static function multiply(Variable $output, \NDArray|float $grad, Variable $a, Variable $b): void
    {
        $a->diff($grad * $b->getArray());
        $b->diff($a->getArray() * $grad);
    }
    public static function matmul(Variable $output, \NDArray|float $grad, Variable $a, Variable $b): void
    {
        $b_transpose = nd::transpose($b->getArray());
        $a_transpose = nd::transpose($a->getArray());
        $dx_da = nd::matmul($grad, $b_transpose);
        $dx_db = nd::matmul($a_transpose, $grad);
        $a->diff($dx_da);
        $b->diff($dx_db);
    }
    public static function clip(Variable $output, \NDArray|float $grad, Variable $a, Variable $min, Variable $max): void
    {
        $greater = nd::greater_equal($a->getArray(), nd::ones($a->getArray()->shape()) * $min->getArray());
        $less = nd::less_equal($a->getArray(), nd::ones($a->getArray()->shape()) * $max->getArray());
        $a->diff($grad * $greater * $less);
    }
    public static function conv2d(Variable $output, \NDArray|float $grad, Variable $input, Variable $filters, array $strides): void
    {
        [$dW, $dInput] = nd::dnn_conv2d_backward($input->getArray(), $grad, $filters->getArray());
        $input->diff($dInput);
        $filters->diff($dW);
    }
    public static function matrix_rank(Variable $output, \NDArray|float $grad, Variable $x): void
    {
        $x->diff(nd::zeros($x->getShape()));
    }
    public static function cond(Variable $output, \NDArray|float $grad, Variable $x): void
    {
        $x->diff(nd::zeros($x->getShape()));
    }
    public static function divide(Variable $output, \NDArray|float $grad, Variable $a, Variable $b): void
    {
        $a->diff($grad / $b->getArray());
        $b->diff(-$grad * $a->getArray() / $b->getArray() ** 2);
    }
    public static function exp(Variable $output, \NDArray|float $grad, Variable $a): void
    {
        $a->diff($grad * nd::exp($a->getArray()));
    }
    public static function exp2(Variable $output, \NDArray|float $grad, Variable $a): void
    {
        $a->diff($grad * $output->getArray() * M_LN2);
    }
    public static function expm1(Variable $output, \NDArray|float $grad, Variable $a): void
    {
        $a->diff($grad * ($output->getArray() + 1));
    }
    public static function mod(Variable $output, \NDArray|float $grad, Variable $x, Variable $y): void
    {
        $x->diff($grad);
        $y->diff(nd::zeros($grad->shape()));
    }
    public static function trunc(Variable $output, \NDArray|float $grad, Variable $a): void
    {
        $a->diff(nd::zeros($grad->shape()));
    }
    public static function floor(Variable $output, \NDArray|float $grad, Variable $a): void
    {
        $a->diff(nd::zeros($grad->shape()));
    }
    public static function sin(Variable $output, \NDArray|float $grad, Variable $a): void
    {
        $a->diff($grad * nd::cos($a->getArray()));
    }
    public static function sinh(Variable $output, \NDArray|float $grad, Variable $a): void
    {
        $a->diff($grad * nd::cosh($a->getArray()));
    }
    public static function ceil(Variable $output, \NDArray|float $grad, Variable $a): void
    {
        $a->diff(nd::zeros($grad->shape()));
    }
    public static function sinc(Variable $output, \NDArray|float $grad, Variable $a): void
    {
        $ppi = $a->getArray() * M_PI;
        $squaredPi = $a->getArray() * $a->getArray() * M_PI;
        $a->diff($grad * (($ppi * nd::cos($ppi) - nd::sin($ppi)) / ($squaredPi)));
    }
    public static function mean(Variable $output, \NDArray|float $grad, Variable $a): void
    {
        $prod = nd::prod(nd::array($a->getArray()->shape()));
        $out = $grad * nd::ones($a->getArray()->shape()) / $prod;
        if ($a->getArray()->isGPU()) {
            $out = $out->gpu();
        }
        $a->diff($out);
    }
    public static function abs(Variable $output, \NDArray|float $grad, Variable $a): void
    {
        $a->diff($grad * nd::sign($a->getArray()));
    }
    public static function acos(Variable $output, \NDArray|float $grad, Variable $a): void
    {
        $a->diff($grad * -(nd::rsqrt(-$a->getArray() * $a->getArray() + 1)));
    }
    public static function cosh(Variable $output, \NDArray|float $grad, Variable $a): void
    {
        $a->diff($grad * nd::sinh($a->getArray()));
    }
    public static function tan(Variable $output, \NDArray|float $grad, Variable $a): void
    {
        $a->diff($grad * (1 + ($output->getArray() ** 2)));
    }
    public static function radians(Variable $output, \NDArray|float $grad, Variable $a): void
    {
        $a->diff($grad * 0.01745329251994329576923690768488612713);
    }
    public static function arccosh(Variable $output, \NDArray|float $grad, Variable $a): void
    {
        $a->diff($grad * nd::rsqrt(($a->getArray() ** 2) - 1));
    }
    public static function arctan(Variable $output, \NDArray|float $grad, Variable $a): void
    {
        $a->diff($grad / ($a->getArray() * $a->getArray() + 1));
    }
    public static function arcsin(Variable $output, \NDArray|float $grad, Variable $a): void
    {
        $a->diff($grad * nd::rsqrt(-$a->getArray() * $a->getArray() + 1));
    }
    public static function arctanh(Variable $output, \NDArray|float $grad, Variable $a): void
    {
        $a->diff($grad * 1 / (1 - ($a->getArray() ** 2)));
    }
    public static function arcsinh(Variable $output, \NDArray|float $grad, Variable $a): void
    {
        $a->diff($grad * nd::rsqrt(($a->getArray() ** 2) + 1));
    }
    public static function cos(Variable $output, \NDArray|float $grad, Variable $a): void
    {
        $a->diff($grad * -nd::sin($a->getArray()));
    }
    public static function rsqrt(Variable $output, \NDArray|float $grad, Variable $a): void
    {
        $a->diff(-0.5 * $grad * ($output->getArray() ** 3));
    }
    public static function reshape(Variable $output, \NDArray|float $grad, Variable $a, array $shape): void
    {
        $a->diff(nd::reshape($grad, $a->getArray()->shape()));
    }
    public static function log(Variable $output, \NDArray|float $grad, Variable $a): void
    {
        $a->diff($grad / $a->getArray());
    }
    public static function log1p(Variable $output, \NDArray|float $grad, Variable $a): void
    {
        $a->diff($grad / ($a->getArray() + 1));
    }
    public static function norm(Variable $output, \NDArray|float $grad, Variable $a): void
    {
        $a->diff($a->getArray() / $output->getArray());
    }
    public static function log2(Variable $output, \NDArray|float $grad, Variable $a): void
    {
        $a->diff($grad / ($a->getArray() * 0.6931471805599453));
    }
    public static function negative(Variable $output, \NDArray|float $grad, Variable $a): void
    {
        $a->diff(-$grad);
    }
    public static function det(Variable $output, \NDArray|float $grad, Variable $a): void
    {
        $a->diff(nd::transpose(nd::inv($a->getArray())) * $output->getArray());
    }
    public static function log10(Variable $output, \NDArray|float $grad, Variable $a): void
    {
        $a->diff($grad / ($a->getArray() * 2.3025850929940456));
    }
    public static function outer(Variable $output, \NDArray|float $grad, Variable $x, Variable $y): void
    {
        $x->diff(nd::sum($y->getArray()) * nd::ones($y->getShape()));
        $y->diff(nd::sum($x->getArray()) * nd::ones($x->getShape()));
    }
    public static function binary_cross_entropy(Variable $output, \NDArray|float $grad, Variable $x, Variable $y, float $epsilon, string $reduction): void
    {
        switch($reduction) {
            case 'mean':
                $result = $grad * ($x->getArray() - $y->getArray()) / nd::clip($x->getArray() * (1 - $x->getArray()), $epsilon, PHP_FLOAT_MAX);
                $result = $result / $x->numElements();
                $grad_target = -(nd::log($x->getArray() / (1 - $x->getArray())));
                $x->diff($result);
                $y->diff($grad_target);
                break;
        }
    }
    public static function sqrt(Variable $output, \NDArray|float $grad, Variable $a): void
    {
        $a->diff($grad / (2 * $output->getArray()));
    }
    public static function tanh(Variable $output, \NDArray|float $grad, Variable $a): void
    {
        $a->diff($grad * (1 - $output->getArray() ** 2));
    }
    public static function sum(Variable $output, \NDArray|float $grad, Variable $a, bool $keepDim): void
    {
        if ($a->isScalar()) {
            $a->diff($grad);
            return;
        }
        $a->diff(nd::ones($a->getArray()->shape()) * $grad);
    }
    public static function sum_axis(Variable $output, \NDArray|float $grad, Variable $a, int $axis, bool $keepDim): void
    {
        $ones = nd::ones($a->getArray()->shape());
        if ($a->getArray()->isGPU()) {
            $ones = $ones->gpu();
        }
        $a->diff($ones * $grad);
    }
    public static function cce(Variable $output, \NDArray|float $grad, Variable $true, Variable $pred, float $epsilon): void
    {
        $pred_batch = nd::clip($pred->getArray(), $epsilon, 1 - $epsilon);
        $dL_dy_pred = -($true->getArray() / $pred_batch);
        $dL_dy_true = -nd::log($pred->getArray());

        $dL_dy_pred = $dL_dy_pred / count($true->getShape());
        $pred->diff($grad * $dL_dy_true);
        $true->diff($grad * $dL_dy_pred);
    }
    public static function relu(Variable $output, \NDArray|float $grad, Variable $a): void
    {
        $a->diff($grad * (nd::greater($a->getArray(), 0)));
    }
    public static function selu(Variable $output, \NDArray|float $grad, Variable $a, float $alpha, float $scale): void
    {
        $non_zero = nd::greater($a->getArray(), 0);
        $zeros = nd::less_equal($a->getArray(), 0);
        $zeros = $zeros * ($alpha * (nd::exp($a->getArray())));

        $a->diff($grad * ($scale * ($non_zero + $zeros)));
    }
    public static function celu(Variable $output, \NDArray|float $grad, Variable $a, float $alpha): void
    {
        $scale = 1.0;
        $negcoef = $alpha * $scale;
        $poscoef = $scale;
        $negiptcoef = $scale / $alpha;

        $zeros = nd::less_equal($a->getArray(), 0);
        $greater = nd::greater($a->getArray(), 0);
        $cond1 = $zeros * ($output->grad() * $negiptcoef * $negcoef * nd::exp($a->getArray()) * $negiptcoef);
        $cond2 = $greater * ($output->grad() * $poscoef);
        $a->diff($cond1 + $cond2);
    }
    public static function power(Variable $output, \NDArray|float $grad, Variable $a, Variable $b): void
    {
        $a->diff($grad * $b->getArray() * $a->getArray() ** ($b->getArray() - 1));
        $b->diff($grad * $a->getArray() ** $b->getArray() * nd::log($a->getArray()));
    }
    public static function dot(Variable $output, \NDArray|float $grad, Variable $a, Variable $b): void
    {
        $a->diff($grad * $b->getArray());
        $b->diff($grad * $a->getArray());
    }
}