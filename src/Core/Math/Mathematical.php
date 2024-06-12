<?php

namespace NumPower\Core\Math;

use NDArray as nd;
use Exception;
use NumPower\Tensor;

trait Mathematical
{
    /**
     * @return \NDArray|float|int
     */
    abstract public function getArray(): \NDArray|float|int;

    /**
     * @return bool
     */
    abstract public function requireGrad(): bool;

    /**
     * @param string $name
     * @return Tensor
     * @throws Exception
     * @throws Exception
     */
    public function rsqrt(string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::rsqrt($this->getArray()), requireGrad: $this->requireGrad());
        $new_var->registerOperation("rsqrt", [$this])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function mean(string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::average($this->getArray()), requireGrad: $this->requireGrad());
        $new_var->registerOperation("mean", [$this])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function abs(string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::abs($this->getArray()), requireGrad: $this->requireGrad());
        $new_var->registerOperation("abs", [$this])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function sqrt(string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::sqrt($this->getArray()), requireGrad: $this->requireGrad());
        $new_var->registerOperation("sqrt", [$this])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function sigmoid(string $name = ''): Tensor
    {
        $ones = new Tensor(1, requireGrad: $this->requireGrad());
        $output = $ones->divide($this->multiply(-1, name: $name)->exp(name: $name)->add($ones, name: $name), name: $name);
        $output->setName($name, $this);
        return $output;
    }
}