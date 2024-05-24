<?php

namespace NumPower\Tensor\Utils;

use Exception;
use NDArray as nd;
use NumPower\Tensor\Variable;

class ValidationUtils
{
    /**
     * @throws Exception
     * @return Variable[]
     */
    public static function validateOperationInputs($name = '', ...$args): array
    {
        $outputs = [];
        foreach ($args as $arg) {
            if (is_array($arg) || is_a($arg, "\gdImage") || is_int($arg) || is_float($arg)) {
                $outputs[] = new Variable(nd::array($arg), name: $arg);
                continue;
            }
            if (is_a($arg, "\NDArray")) {
                $outputs[] = new Variable($arg, name: $name);
                continue;
            }
            if (is_a($arg, Variable::class)) {
                $outputs[] = $arg;
                continue;
            }
            throw new Exception("Invalid input for operation.");
        }
        return $outputs;
    }
}