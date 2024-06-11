<?php

namespace NumPower\Tensor\Utils;

use Exception;
use NDArray as nd;
use NumPower\Tensor\Tensor;

class ValidationUtils
{
    /**
     * @return Tensor[]
     *@throws Exception
     */
    public static function validateOperationInputs($name = '', ...$args): array
    {
        $outputs = [];
        foreach ($args as $arg) {
            if (is_array($arg) || is_a($arg, "\gdImage") || is_int($arg) || is_float($arg)) {
                $outputs[] = new Tensor(nd::array($arg), name: $arg);
                continue;
            }
            if (is_a($arg, "\NDArray")) {
                $outputs[] = new Tensor($arg, name: $name);
                continue;
            }
            if (is_a($arg, Tensor::class)) {
                $outputs[] = $arg;
                continue;
            }
            throw new Exception("Invalid input for operation.");
        }
        return $outputs;
    }
}