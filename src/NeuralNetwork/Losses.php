<?php

namespace NumPower\Tensor\NeuralNetwork;

use Exception;
use NumPower\Tensor\Utils\ValidationUtils;
use NumPower\Tensor\Variable;
use NDArray as nd;

class Losses
{
    /**
     * @param int|float|array|object $x
     * @param int|float|array|object $y
     * @param string|null $reduction
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public static function MeanSquaredError(int|float|array|object $x,
                                            int|float|array|object $y,
                                            ?string                $reduction = 'mean',
                                            string                 $name = ''): Variable
    {
        [$x, $y] = ValidationUtils::validateOperationInputs($name, $x, $y);
        $loss = $x->subtract($y, name: $name)->power(2, name: $name);
        if (isset($reduction) && $reduction != '') {
            $loss = $loss->{$reduction}(name: $name);
        }
        $loss->setName($name);
        return $loss;
    }

    /**
     * @param int|float|array|object $x
     * @param int|float|array|object $y
     * @param string|null $reduction
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public static function MeanAbsoluteError(int|float|array|object $x,
                                             int|float|array|object $y,
                                             ?string                $reduction = 'mean',
                                             string                 $name = ''): Variable
    {
        [$x, $y] = ValidationUtils::validateOperationInputs($name, $x, $y);
        $loss = $x->subtract($y, name: $name)->abs(name: $name);

        if (isset($reduction) && $reduction != '') {
            $loss = $loss->{$reduction}();
        }

        $loss->setName($name);
        return $loss;
    }


    /**
     * @param int|float|array|object $x
     * @param int|float|array|object $y
     * @param float $epsilon
     * @param string|null $reduction
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public static function BinaryCrossEntropy(int|float|array|object $x,
                                              int|float|array|object $y,
                                              float                  $epsilon = 1e-15,
                                              ?string                $reduction = 'mean',
                                              string                 $name = ''): Variable
    {
        [$x, $y] = ValidationUtils::validateOperationInputs($name, $x, $y);
        $loss = ($y->getArray() - 1) *
            nd::maximum(nd::log1p(-$x->getArray()), nd::ones($x->getShape()) * -100) -
            $y->getArray() *
            nd::maximum(nd::log($x->getArray()), (nd::ones($x->getShape()) * -100));

        $new_var = new Variable($loss);
        if (isset($reduction) && $reduction != '') {
            $new_var = new Variable(nd::{$reduction}($loss));
        }
        $new_var->registerOperation('binary_cross_entropy', [$x, $y, $epsilon, $reduction])->setName($name);
        return $new_var;
    }
}