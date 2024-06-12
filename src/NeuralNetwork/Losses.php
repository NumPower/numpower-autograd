<?php

namespace NumPower\NeuralNetwork;

use Exception;
use NumPower\Utils\ValidationUtils;
use NumPower\Tensor;
use NDArray as nd;

class Losses
{
    /**
     * @param int|float|array|object $x
     * @param int|float|array|object $y
     * @param string|null $reduction
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public static function MeanSquaredError(int|float|array|object $x,
                                            int|float|array|object $y,
                                            ?string                $reduction = 'mean',
                                            string                 $name = ''): Tensor
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
     * @return Tensor
     * @throws Exception
     */
    public static function MeanAbsoluteError(int|float|array|object $x,
                                             int|float|array|object $y,
                                             ?string                $reduction = 'mean',
                                             string                 $name = ''): Tensor
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
     * @return Tensor
     * @throws Exception
     */
    public static function BinaryCrossEntropy(int|float|array|object $x,
                                              int|float|array|object $y,
                                              float                  $epsilon = 1e-15,
                                              ?string                $reduction = 'mean',
                                              string                 $name = ''): Tensor
    {
        [$x, $y] = ValidationUtils::validateOperationInputs($name, $x, $y);
        $loss = ($y->getArray() - 1) *
            nd::maximum(nd::log1p(-$x->getArray()), nd::ones($x->getShape()) * -100) -
            $y->getArray() *
            nd::maximum(nd::log($x->getArray()), (nd::ones($x->getShape()) * -100));

        if (isset($reduction) && $reduction != '') {
            $new_var = new Tensor(nd::{$reduction}($loss), requireGrad: $x->requireGrad());
        } else {
            $new_var = new Tensor($loss, requireGrad: $x->requireGrad());
        }
        $new_var->registerOperation('binary_cross_entropy', [$x, $y, $epsilon, $reduction])->setName($name);
        return $new_var;
    }
}