<?php

namespace NumPower\Tensor\Core\Math;

use Exception;
use NDArray as nd;
use NumPower\Tensor\Utils\ValidationUtils;
use NumPower\Tensor\Tensor;

trait Rounding
{
    /**
     * @return nd|float|int
     */
    abstract public function getArray(): nd|float|int;

    /**
     * @param float $min
     * @param float $max
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function clip(float $min, float $max, string $name = ''): Tensor
    {
        [$input_min, $input_max] = ValidationUtils::validateOperationInputs($name, $min, $max);
        $output = new Tensor(nd::clip($this->getArray(), $min, $max));
        $output->registerOperation("clip", [$this, $input_min, $input_max])->setName($name, $this);
        return $output;
    }

    /**
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function trunc(string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::trunc($this->getArray()));
        $new_var->registerOperation("trunc", [$this])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function floor(string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::floor($this->getArray()));
        $new_var->registerOperation("floor", [$this])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function ceil(string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::ceil($this->getArray()));
        $new_var->registerOperation("ceil", [$this])->setName($name, $this);
        return $new_var;
    }
}