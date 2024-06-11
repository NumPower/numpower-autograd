<?php

namespace NumPower\Tensor\Core\Tape;

use Exception;
use NDArray as nd;
use NumPower\Tensor\Tensor;

/**
 * Core backward functions.
 *
 * Implements a static function named after all core operations to calculate
 * partial derivatives.
 */
class BackwardOperation
{
    public static function offsetGet(Tensor $output, \NDArray|float $grad, Tensor $self, int $offset): void
    {
        $zeros = nd::zeros($self->getShape());
        if (is_float($grad) || count($grad->shape()) < count($zeros->shape())) {
            $zeros[$offset] = $grad;
        } else if (count($grad->shape()) > count($zeros->shape())) {
            $zeros[$offset] = nd::sum($grad);
        }

        $self->diff($zeros);
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @param Tensor $b
     * @return void
     * @throws Exception
     */
    public static function add(Tensor $output, \NDArray|float $grad, Tensor $a, Tensor $b): void
    {
        $a->diff($grad);
        $b->diff($grad);
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @param Tensor $b
     * @return void
     * @throws Exception
     */
    public static function subtract(Tensor $output, \NDArray|float $grad, Tensor $a, Tensor $b): void
    {
        $a->diff($grad);
        $b->diff(-$grad);
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @param Tensor $b
     * @return void
     * @throws Exception
     */
    public static function multiply(Tensor $output, \NDArray|float $grad, Tensor $a, Tensor $b): void
    {
        $a->diff($grad * $b->getArray());
        $b->diff($a->getArray() * $grad);
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @param Tensor $b
     * @return void
     * @throws Exception
     */
    public static function matmul(Tensor $output, \NDArray|float $grad, Tensor $a, Tensor $b): void
    {
        $b_transpose = nd::transpose($b->getArray());
        $a_transpose = nd::transpose($a->getArray());
        $dx_da = nd::matmul($grad, $b_transpose);
        $dx_db = nd::matmul($a_transpose, $grad);
        $a->diff($dx_da);
        $b->diff($dx_db);
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @param Tensor $min
     * @param Tensor $max
     * @return void
     * @throws Exception
     */
    public static function clip(Tensor $output, \NDArray|float $grad, Tensor $a, Tensor $min, Tensor $max): void
    {
        $greater = nd::greater_equal($a->getArray(), nd::ones($a->getArray()->shape()) * $min->getArray());
        $less = nd::less_equal($a->getArray(), nd::ones($a->getArray()->shape()) * $max->getArray());
        $a->diff($grad * $greater * $less);
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $x
     * @return void
     * @throws Exception
     */
    public static function matrix_rank(Tensor $output, \NDArray|float $grad, Tensor $x): void
    {
        $x->diff(nd::zeros($x->getShape()));
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $x
     * @return void
     * @throws Exception
     */
    public static function cond(Tensor $output, \NDArray|float $grad, Tensor $x): void
    {
        $x->diff(nd::zeros($x->getShape()));
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @param Tensor $b
     * @return void
     * @throws Exception
     */
    public static function divide(Tensor $output, \NDArray|float $grad, Tensor $a, Tensor $b): void
    {
        $a->diff($grad / $b->getArray());
        $b->diff(-$grad * $a->getArray() / $b->getArray() ** 2);
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @return void
     * @throws Exception
     */
    public static function exp(Tensor $output, \NDArray|float $grad, Tensor $a): void
    {
        $a->diff($grad * nd::exp($a->getArray()));
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @return void
     * @throws Exception
     */
    public static function exp2(Tensor $output, \NDArray|float $grad, Tensor $a): void
    {
        $a->diff($grad * $output->getArray() * M_LN2);
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @return void
     * @throws Exception
     */
    public static function expm1(Tensor $output, \NDArray|float $grad, Tensor $a): void
    {
        $a->diff($grad * ($output->getArray() + 1));
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $x
     * @param Tensor $y
     * @return void
     * @throws Exception
     */
    public static function mod(Tensor $output, \NDArray|float $grad, Tensor $x, Tensor $y): void
    {
        $x->diff($grad);
        $y->diff(nd::zeros($grad->shape()));
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @return void
     * @throws Exception
     */
    public static function trunc(Tensor $output, \NDArray|float $grad, Tensor $a): void
    {
        $a->diff(nd::zeros($grad->shape()));
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @return void
     * @throws Exception
     */
    public static function floor(Tensor $output, \NDArray|float $grad, Tensor $a): void
    {
        $a->diff(nd::zeros($grad->shape()));
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @return void
     * @throws Exception
     */
    public static function sin(Tensor $output, \NDArray|float $grad, Tensor $a): void
    {
        $a->diff($grad * nd::cos($a->getArray()));
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @return void
     * @throws Exception
     */
    public static function sinh(Tensor $output, \NDArray|float $grad, Tensor $a): void
    {
        $a->diff($grad * nd::cosh($a->getArray()));
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @return void
     * @throws Exception
     */
    public static function ceil(Tensor $output, \NDArray|float $grad, Tensor $a): void
    {
        $a->diff(nd::zeros($grad->shape()));
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @return void
     * @throws Exception
     */
    public static function sinc(Tensor $output, \NDArray|float $grad, Tensor $a): void
    {
        $ppi = $a->getArray() * M_PI;
        $squaredPi = $a->getArray() * $a->getArray() * M_PI;
        $a->diff($grad * (($ppi * nd::cos($ppi) - nd::sin($ppi)) / ($squaredPi)));
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @return void
     * @throws Exception
     */
    public static function mean(Tensor $output, \NDArray|float $grad, Tensor $a): void
    {
        $prod = nd::prod(nd::array($a->getArray()->shape()));
        $out = $grad * nd::ones($a->getArray()->shape()) / $prod;
        if ($a->getArray()->isGPU()) {
            $out = $out->gpu();
        }
        $a->diff($out);
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @return void
     * @throws Exception
     */
    public static function abs(Tensor $output, \NDArray|float $grad, Tensor $a): void
    {
        $a->diff($grad * nd::sign($a->getArray()));
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @return void
     * @throws Exception
     */
    public static function acos(Tensor $output, \NDArray|float $grad, Tensor $a): void
    {
        $a->diff($grad * -(nd::rsqrt(-$a->getArray() * $a->getArray() + 1)));
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @return void
     * @throws Exception
     */
    public static function cosh(Tensor $output, \NDArray|float $grad, Tensor $a): void
    {
        $a->diff($grad * nd::sinh($a->getArray()));
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @return void
     * @throws Exception
     */
    public static function tan(Tensor $output, \NDArray|float $grad, Tensor $a): void
    {
        $a->diff($grad * (1 + ($output->getArray() ** 2)));
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @return void
     * @throws Exception
     */
    public static function radians(Tensor $output, \NDArray|float $grad, Tensor $a): void
    {
        $a->diff($grad * 0.01745329251994329576923690768488612713);
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @return void
     * @throws Exception
     */
    public static function arccosh(Tensor $output, \NDArray|float $grad, Tensor $a): void
    {
        $a->diff($grad * nd::rsqrt(($a->getArray() ** 2) - 1));
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @return void
     * @throws Exception
     */
    public static function arctan(Tensor $output, \NDArray|float $grad, Tensor $a): void
    {
        $a->diff($grad / ($a->getArray() * $a->getArray() + 1));
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @return void
     * @throws Exception
     */
    public static function arcsin(Tensor $output, \NDArray|float $grad, Tensor $a): void
    {
        $a->diff($grad * nd::rsqrt(-$a->getArray() * $a->getArray() + 1));
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @return void
     * @throws Exception
     */
    public static function arctanh(Tensor $output, \NDArray|float $grad, Tensor $a): void
    {
        $a->diff($grad * 1 / (1 - ($a->getArray() ** 2)));
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @return void
     * @throws Exception
     */
    public static function arcsinh(Tensor $output, \NDArray|float $grad, Tensor $a): void
    {
        $a->diff($grad * nd::rsqrt(($a->getArray() ** 2) + 1));
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @return void
     * @throws Exception
     */
    public static function cos(Tensor $output, \NDArray|float $grad, Tensor $a): void
    {
        $a->diff($grad * -nd::sin($a->getArray()));
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @return void
     * @throws Exception
     */
    public static function rsqrt(Tensor $output, \NDArray|float $grad, Tensor $a): void
    {
        $a->diff(-0.5 * $grad * ($output->getArray() ** 3));
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @param array $shape
     * @return void
     * @throws Exception
     */
    public static function reshape(Tensor $output, \NDArray|float $grad, Tensor $a, array $shape): void
    {
        $a->diff(nd::reshape($grad, $a->getArray()->shape()));
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @return void
     * @throws Exception
     */
    public static function log(Tensor $output, \NDArray|float $grad, Tensor $a): void
    {
        $a->diff($grad / $a->getArray());
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @return void
     * @throws Exception
     */
    public static function log1p(Tensor $output, \NDArray|float $grad, Tensor $a): void
    {
        $a->diff($grad / ($a->getArray() + 1));
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @return void
     * @throws Exception
     */
    public static function norm(Tensor $output, \NDArray|float $grad, Tensor $a): void
    {
        $a->diff($a->getArray() / $output->getArray());
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @return void
     * @throws Exception
     */
    public static function log2(Tensor $output, \NDArray|float $grad, Tensor $a): void
    {
        $a->diff($grad / ($a->getArray() * 0.6931471805599453));
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @return void
     * @throws Exception
     */
    public static function negative(Tensor $output, \NDArray|float $grad, Tensor $a): void
    {
        $a->diff(-$grad);
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @return void
     * @throws Exception
     */
    public static function det(Tensor $output, \NDArray|float $grad, Tensor $a): void
    {
        $a->diff(nd::transpose(nd::inv($a->getArray())) * $output->getArray());
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @return void
     * @throws Exception
     */
    public static function log10(Tensor $output, \NDArray|float $grad, Tensor $a): void
    {
        $a->diff($grad / ($a->getArray() * 2.3025850929940456));
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $x
     * @param Tensor $y
     * @return void
     * @throws Exception
     */
    public static function outer(Tensor $output, \NDArray|float $grad, Tensor $x, Tensor $y): void
    {
        $x->diff(nd::sum($y->getArray()) * nd::ones($y->getShape()));
        $y->diff(nd::sum($x->getArray()) * nd::ones($x->getShape()));
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $x
     * @param Tensor $y
     * @param float $epsilon
     * @param string $reduction
     * @return void
     * @throws Exception
     */
    public static function binary_cross_entropy(Tensor $output, \NDArray|float $grad, Tensor $x, Tensor $y, float $epsilon, string $reduction): void
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

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @return void
     * @throws Exception
     */
    public static function sqrt(Tensor $output, \NDArray|float $grad, Tensor $a): void
    {
        $a->diff($grad / (2 * $output->getArray()));
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @return void
     * @throws Exception
     */
    public static function tanh(Tensor $output, \NDArray|float $grad, Tensor $a): void
    {
        $a->diff($grad * (1 - $output->getArray() ** 2));
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @param bool $keepDim
     * @return void
     * @throws Exception
     */
    public static function sum(Tensor $output, \NDArray|float $grad, Tensor $a, bool $keepDim): void
    {
        if ($a->isScalar()) {
            $a->diff($grad);
            return;
        }
        $a->diff(nd::ones($a->getArray()->shape()) * $grad);
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @param int $axis
     * @param bool $keepDim
     * @return void
     * @throws Exception
     */
    public static function sum_axis(Tensor $output, \NDArray|float $grad, Tensor $a, int $axis, bool $keepDim): void
    {
        $ones = nd::ones($a->getArray()->shape());
        if ($a->getArray()->isGPU()) {
            $ones = $ones->gpu();
        }
        $a->diff($ones * $grad);
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $true
     * @param Tensor $pred
     * @param float $epsilon
     * @return void
     * @throws Exception
     */
    public static function cce(Tensor $output, \NDArray|float $grad, Tensor $true, Tensor $pred, float $epsilon): void
    {
        $pred_batch = nd::clip($pred->getArray(), $epsilon, 1 - $epsilon);
        $dL_dy_pred = -($true->getArray() / $pred_batch);
        $dL_dy_true = -nd::log($pred->getArray());

        $dL_dy_pred = $dL_dy_pred / count($true->getShape());
        $pred->diff($grad * $dL_dy_true);
        $true->diff($grad * $dL_dy_pred);
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @return void
     * @throws Exception
     */
    public static function relu(Tensor $output, \NDArray|float $grad, Tensor $a): void
    {
        $a->diff($grad * (1 * nd::greater($a->getArray(), 0)));
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @param float $alpha
     * @param float $scale
     * @return void
     * @throws Exception
     */
    public static function selu(Tensor $output, \NDArray|float $grad, Tensor $a, float $alpha, float $scale): void
    {
        $non_zero = nd::greater($a->getArray(), 0);
        $zeros = nd::less_equal($a->getArray(), 0);
        $zeros = $zeros * ($alpha * (nd::exp($a->getArray())));

        $a->diff($grad * ($scale * ($non_zero + $zeros)));
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @param float $alpha
     * @return void
     * @throws Exception
     */
    public static function celu(Tensor $output, \NDArray|float $grad, Tensor $a, float $alpha): void
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

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @param Tensor $b
     * @return void
     * @throws Exception
     */
    public static function power(Tensor $output, \NDArray|float $grad, Tensor $a, Tensor $b): void
    {
        $a->diff($grad * $b->getArray() * $a->getArray() ** ($b->getArray() - 1));
        $b->diff($grad * $a->getArray() ** $b->getArray() * nd::log($a->getArray()));
    }

    /**
     * @param Tensor $output
     * @param nd|float $grad
     * @param Tensor $a
     * @param Tensor $b
     * @return void
     * @throws Exception
     */
    public static function dot(Tensor $output, \NDArray|float $grad, Tensor $a, Tensor $b): void
    {
        $a->diff($grad * $b->getArray());
        $b->diff($grad * $a->getArray());
    }
}