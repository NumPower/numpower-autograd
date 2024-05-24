<?php

namespace NumPower\Tensor\Core\Math;

use Exception;
use NumPower\Tensor\Variable;

trait Trigonometrics
{
    /**
     * @return \NDArray|float|int
     */
    abstract public function getArray(): \NDArray|float|int;

    /**
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function acos(string $name = ''): Variable
    {
        $new_var = new Variable(nd::arccos($this->getArray()));
        $new_var->registerOperation("acos", [$this])->setName($name);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function arcsin(string $name = ''): Variable
    {
        $new_var = new Variable(nd::arcsin($this->getArray()));
        $new_var->registerOperation("arcsin", [$this])->setName($name);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function sin(string $name = ''): Variable
    {
        $new_var = new Variable(nd::sin($this->getArray()));
        $new_var->registerOperation("sin", [$this])->setName($name);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function radians(string $name = ''): Variable
    {
        $new_var = new Variable(nd::radians($this->getArray()));
        $new_var->registerOperation("radians", [$this])->setName($name);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function cos(string $name = ''): Variable
    {
        $new_var = new Variable(nd::cos($this->getArray()));
        $new_var->registerOperation("cos", [$this])->setName($name);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function arctan(string $name = ''): Variable
    {
        $new_var = new Variable(nd::arctan($this->getArray()));
        $new_var->registerOperation("arctan", [$this])->setName($name);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function tan(string $name = ''): Variable
    {
        $new_var = new Variable(nd::tan($this->getArray()));
        $new_var->registerOperation("tan", [$this])->setName($name);
        return $new_var;
    }
}