<?php

namespace NumPower\Core\Math;

use Exception;
use NDArray as nd;
use NumPower\Tensor;

trait Rounding
{
    /**
     * @return nd|float|int
     */
    abstract public function getArray(): nd|float|int;

    /**
     * @return bool
     */
    abstract public function requireGrad(): bool;

    /**
     * @param float $min
     * @param float $max
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function clip(float $min, float $max, string $name = ''): Tensor
    {
        $output = new Tensor(nd::clip($this->getArray(), $min, $max), requireGrad: $this->requireGrad());
        $output->registerOperation("clip", [$this, $min, $max])->setName($name, $this);
        return $output;
    }

    /**
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function trunc(string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::trunc($this->getArray()), requireGrad: $this->requireGrad());
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
        $new_var = new Tensor(nd::floor($this->getArray()), requireGrad: $this->requireGrad());
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
        $new_var = new Tensor(nd::ceil($this->getArray()), requireGrad: $this->requireGrad());
        $new_var->registerOperation("ceil", [$this])->setName($name, $this);
        return $new_var;
    }
}