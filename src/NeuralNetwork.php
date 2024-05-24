<?php

namespace NumPower\Tensor;

use Exception;
use NumPower\Tensor\Utils\ValidationUtils;
use NDArray as nd;

class NeuralNetwork
{
    /**
     * @param $input
     * @param $filters
     * @param $strides
     * @param $padding
     * @param $dataFormat
     * @param $dilations
     * @return Variable
     * @throws Exception
     */
    public static function conv2d($input, $filters, $strides, $padding, $dilations = null, $dataFormat = 'NCHW'): Variable
    {
        if (!is_a($input, Variable::class)) {
            $input = new Variable(ValidationUtils::validateOperationInputs($input)[0]);
        }
        if (!is_a($filters, Variable::class)) {
            $filters = new Variable(ValidationUtils::validateOperationInputs($filters)[0]);
        }

        $output = new Variable(nd::dnn_conv2d_forward($input->getArray(), $filters->getArray()));
        $output->registerOperation("conv2d", [$input, $filters, $strides, $padding, $dilations, $dataFormat]);
        return $output;
    }

    /**
     * Categorical Cross Entropy
     *
     * @param Variable $true
     * @param Variable $pred
     * @param float $epsilon
     * @return Variable
     * @throws Exception
     */
    public static function cce(Variable $true, Variable $pred, float $epsilon = 1e-15): Variable
    {
        $output = $pred->divide($pred->sum_axis(axis: 1, keepdim: true));
        $out = $output->clip($epsilon, 1 - $epsilon);
        $m_ones = nd::ones($pred->getArray()->shape()) * -1;
        $out = $true->multiply($out->log())->sum()->multiply($m_ones)->divide(count($true->getArray()));
        $output = new Variable($out);
        $output->registerOperation("cce", [$true, $pred, $epsilon]);
        return $output;
    }

    /**
     * @param Variable $inputs
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public static function ReLU(Variable $inputs, string $name = ''): Variable
    {
        $new_var = new Variable(nd::maximum($inputs->getArray(), 0));
        $new_var->registerOperation("relu", [$inputs])->setName($name);
        return $new_var;
    }

    /**
     * @param Variable $inputs
     * @param float $alpha
     * @param float $scale
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public static function SELU(Variable $inputs, float $alpha=1.67326, float $scale=1.0507, string $name = ''): Variable
    {
        $non_zero = nd::greater($inputs->getArray(), 0);
        $zeros = nd::less_equal($inputs->getArray(), 0);
        $non_zero = $non_zero * $inputs->getArray();
        $zeros = $zeros * ($alpha * (nd::exp($inputs->getArray()) - 1));
        $new_var = new Variable($scale * ($non_zero + $zeros));
        $new_var->registerOperation("selu", [$inputs, $alpha, $scale])->setName($name);
        return $new_var;
    }

    /**
     * @param int|float|array|object $x
     * @param float $alpha
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public static function CELU(int|float|array|object $x,
                                float $alpha = 1.0,
                                string                 $name = ''): Variable
    {
        [$x] = ValidationUtils::validateOperationInputs($name, $x);
        $loss = new Variable(nd::maximum(0, $x->getArray()) + nd::minimum(0, $alpha * (nd::exp($x->getArray() / $alpha) - 1)));
        $loss->registerOperation('celu', [$x, $alpha])->setName($name);
        return $loss;
    }

    /**
     * @param int|float|array|object $x
     * @param float $beta
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public static function SiLU(int|float|array|object $x,
                                float $beta = 1.0,
                                string                 $name = ''): Variable
    {
        [$x, $beta] = ValidationUtils::validateOperationInputs($name, $x, $beta);
        $loss = $beta->multiply($x, name: $name)
            ->sigmoid(name: $name)
            ->multiply($x, name: $name)
            ->setName($name);
        return $loss;
    }
}