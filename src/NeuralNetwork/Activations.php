<?php

namespace NumPower\Tensor\NeuralNetwork;

use Exception;
use \NDArray as nd;
use NumPower\Tensor\Utils\ValidationUtils;
use NumPower\Tensor\Variable;

class Activations
{
    /**
     * @param Variable $inputs
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public static function ReLU(Variable $inputs, string $name = 'out_relu'): Variable
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
    public static function SELU(Variable $inputs, float $alpha=1.67326, float $scale=1.0507, string $name = 'out_selu'): Variable
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
                                string                 $name = 'out_celu'): Variable
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
                                string                 $name = 'out_silu'): Variable
    {
        [$x, $beta] = ValidationUtils::validateOperationInputs($name, $x, $beta);
        $loss = $beta->multiply($x, name: $name)
            ->sigmoid(name: $name)
            ->multiply($x, name: $name)
            ->setName($name, $x);
        return $loss;
    }

    /**
     * @param int|float|array|object $x
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public static function sigmoid(int|float|array|object $x,  string $name = 'out_sigmoid'): Variable
    {
        [$x] = ValidationUtils::validateOperationInputs($name, $x);
        return $x->sigmoid()->setName($name, $x);
    }

    /**
     * @param int|float|array|object $x
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public static function softsign(int|float|array|object $x,  string $name = 'out_sigmoid'): Variable
    {
        [$x] = ValidationUtils::validateOperationInputs($name, $x);
        return $x / $x->abs()->add(1);
    }

    /**
     * @param int|float|array|object $x
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public static function softmax(int|float|array|object $x, string $name = ''): Variable
    {
        [$x] = ValidationUtils::validateOperationInputs($name, $x);
        $output = $x->exp() / $x->exp()->sum_axis(0);
        return $output->setName($name, $x);
    }

    /**
     * @param int|float|array|object $x
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public static function softplus(int|float|array|object $x, string $name = ''): Variable
    {
        [$x] = ValidationUtils::validateOperationInputs($name, $x);
        return $x->exp()->add(1)->log();
    }

    /**
     * @param int|float|array|object $x
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public static function exponential(int|float|array|object $x, string $name = ''): Variable
    {
        [$x] = ValidationUtils::validateOperationInputs($name, $x);
        return $x->exp();
    }

    /**
     * @param int|float|array|object $x
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public static function linear(int|float|array|object $x, string $name = ''): Variable
    {
        [$x] = ValidationUtils::validateOperationInputs($name, $x);
        return $x;
    }

    /**
     * @param int|float|array|object $x
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public static function mish(int|float|array|object $x, string $name = ''): Variable
    {
        [$x] = ValidationUtils::validateOperationInputs($name, $x);
        return $x * $x->exp()->add(1)->log()->tanh();
    }
}