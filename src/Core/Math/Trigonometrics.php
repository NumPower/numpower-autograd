<?php

namespace NumPower\Core\Math;

use Exception;
use NumPower\Tensor;

trait Trigonometrics
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
     */
    public function acos(string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::arccos($this->getArray()), requireGrad: $this->requireGrad());
        $new_var->registerOperation("acos", [$this])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function arcsin(string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::arcsin($this->getArray()), requireGrad: $this->requireGrad());
        $new_var->registerOperation("arcsin", [$this])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function sin(string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::sin($this->getArray()), requireGrad: $this->requireGrad());
        $new_var->registerOperation("sin", [$this])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function radians(string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::radians($this->getArray()), requireGrad: $this->requireGrad());
        $new_var->registerOperation("radians", [$this])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function cos(string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::cos($this->getArray()), requireGrad: $this->requireGrad());
        $new_var->registerOperation("cos", [$this])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function arctan(string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::arctan($this->getArray()), requireGrad: $this->requireGrad());
        $new_var->registerOperation("arctan", [$this])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function tan(string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::tan($this->getArray()), requireGrad: $this->requireGrad());
        $new_var->registerOperation("tan", [$this])->setName($name, $this);
        return $new_var;
    }
}