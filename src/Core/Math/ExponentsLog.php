<?php

namespace NumPower\Tensor\Core\Math;

use NDArray as nd;
use Exception;
use NumPower\Tensor\Variable;

trait ExponentsLog
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
    public function exp(string $name = ''): Variable
    {
        $new_var = new Variable(nd::exp($this->getArray()));
        $new_var->registerOperation("exp", [$this])->setName($name);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function exp2(string $name = ''): Variable
    {
        $new_var = new Variable(nd::exp2($this->getArray()));
        $new_var->registerOperation("exp2", [$this])->setName($name);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function expm1(string $name = ''): Variable
    {
        $new_var = new Variable(nd::expm1($this->getArray()));
        $new_var->registerOperation("expm1", [$this])->setName($name);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function log(string $name = ''): Variable
    {
        $new_var = new Variable(nd::log($this->getArray()));
        $new_var->registerOperation("log", [$this])->setName($name);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function log1p(string $name = ''): Variable
    {
        $new_var = new Variable(nd::log1p($this->getArray()));
        $new_var->registerOperation("log1p", [$this])->setName($name);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function log2(string $name = ''): Variable
    {
        $new_var = new Variable(nd::log2($this->getArray()));
        $new_var->registerOperation("log2", [$this])->setName($name);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function log10(string $name = ''): Variable
    {
        $new_var = new Variable(nd::log10($this->getArray()));
        $new_var->registerOperation("log10", [$this])->setName($name);
        return $new_var;
    }
}