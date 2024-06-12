<?php

namespace NumPower\NeuralNetwork;

use Exception;
use \NDArray as nd;
use NumPower\Utils\ValidationUtils;
use NumPower\Tensor;

class Activations
{
    /**
     * @param Tensor $inputs
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public static function ReLU(Tensor $inputs, string $name = 'out_relu'): Tensor
    {
        $new_var = new Tensor(
            $inputs->getArray() * nd::greater($inputs->getArray(), 0),
            requireGrad: $inputs->requireGrad()
        );
        $new_var->registerOperation("relu", [$inputs])->setName($name);
        return $new_var;
    }

    /**
     * @param Tensor $inputs
     * @param float $alpha
     * @param float $scale
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public static function SELU(Tensor $inputs, float $alpha=1.67326, float $scale=1.0507, string $name = 'out_selu'): Tensor
    {
        $non_zero = nd::greater($inputs->getArray(), 0);
        $zeros = nd::less_equal($inputs->getArray(), 0);
        $non_zero = $non_zero * $inputs->getArray();
        $zeros = $zeros * ($alpha * (nd::exp($inputs->getArray()) - 1));
        $new_var = new Tensor($scale * ($non_zero + $zeros), requireGrad: $inputs->requireGrad());
        $new_var->registerOperation("selu", [$inputs, $alpha, $scale])->setName($name);
        return $new_var;
    }

    /**
     * @param int|float|array|object $x
     * @param float $alpha
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public static function CELU(int|float|array|object $x,
                                float $alpha = 1.0,
                                string                 $name = 'out_celu'): Tensor
    {
        [$x] = ValidationUtils::validateOperationInputs($name, $x);
        $loss = new Tensor(nd::maximum(0, $x->getArray()) + nd::minimum(0, $alpha * (nd::exp($x->getArray() / $alpha) - 1)), requireGrad: $x->requireGrad());
        $loss->registerOperation('celu', [$x, $alpha])->setName($name);
        return $loss;
    }

    /**
     * @param int|float|array|object $x
     * @param float $beta
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public static function SiLU(int|float|array|object $x,
                                float $beta = 1.0,
                                string                 $name = 'out_silu'): Tensor
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
     * @return Tensor
     * @throws Exception
     */
    public static function sigmoid(int|float|array|object $x,  string $name = 'out_sigmoid'): Tensor
    {
        [$x] = ValidationUtils::validateOperationInputs($name, $x);
        return $x->sigmoid()->setName($name, $x);
    }

    /**
     * @param int|float|array|object $x
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public static function softsign(int|float|array|object $x,  string $name = 'out_sigmoid'): Tensor
    {
        [$x] = ValidationUtils::validateOperationInputs($name, $x);
        return $x / $x->abs()->add(1);
    }

    /**
     * @param int|float|array|object $x
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public static function softmax(int|float|array|object $x, string $name = ''): Tensor
    {
        [$x] = ValidationUtils::validateOperationInputs($name, $x);
        $output = $x->exp() / $x->exp()->sum_axis(0);
        return $output->setName($name, $x);
    }

    /**
     * @param int|float|array|object $x
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public static function softplus(int|float|array|object $x, string $name = ''): Tensor
    {
        [$x] = ValidationUtils::validateOperationInputs($name, $x);
        return $x->exp()->add(1)->log();
    }

    /**
     * @param int|float|array|object $x
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public static function exponential(int|float|array|object $x, string $name = ''): Tensor
    {
        [$x] = ValidationUtils::validateOperationInputs($name, $x);
        return $x->exp();
    }

    /**
     * @param int|float|array|object $x
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public static function linear(int|float|array|object $x, string $name = ''): Tensor
    {
        [$x] = ValidationUtils::validateOperationInputs($name, $x);
        return $x;
    }

    /**
     * @param int|float|array|object $x
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public static function mish(int|float|array|object $x, string $name = ''): Tensor
    {
        [$x] = ValidationUtils::validateOperationInputs($name, $x);
        return $x * $x->exp()->add(1)->log()->tanh();
    }
}