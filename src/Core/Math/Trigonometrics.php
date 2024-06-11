<?php

namespace NumPower\Tensor\Core\Math;

use Exception;
use NumPower\Tensor\Tensor;

trait Trigonometrics
{
    /**
     * @return \NDArray|float|int
     */
    abstract public function getArray(): \NDArray|float|int;

    /**
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function acos(string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::arccos($this->getArray()));
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
        $new_var = new Tensor(nd::arcsin($this->getArray()));
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
        $new_var = new Tensor(nd::sin($this->getArray()));
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
        $new_var = new Tensor(nd::radians($this->getArray()));
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
        $new_var = new Tensor(nd::cos($this->getArray()));
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
        $new_var = new Tensor(nd::arctan($this->getArray()));
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
        $new_var = new Tensor(nd::tan($this->getArray()));
        $new_var->registerOperation("tan", [$this])->setName($name, $this);
        return $new_var;
    }
}