<?php

namespace NumPower\Tensor\Core\Math;

use \NDArray as nd;
use Exception;
use NumPower\Tensor\Variable;

trait Hyperbolics
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
    public function arccosh(string $name = ''): Variable
    {
        $new_var = new Variable(nd::arccosh($this->getArray()));
        $new_var->registerOperation("arccosh", [$this])->setName($name);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function sinh(string $name = ''): Variable
    {
        $new_var = new Variable(nd::sinh($this->getArray()));
        $new_var->registerOperation("sinh", [$this])->setName($name);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function tanh(string $name = ''): Variable
    {
        $new_var = new Variable(nd::tanh($this->getArray()));
        $new_var->registerOperation("tanh", [$this])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function cosh(string $name = ''): Variable
    {
        $new_var = new Variable(nd::cosh($this->getArray()));
        $new_var->registerOperation("cosh", [$this])->setName($name);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function arcsinh(string $name = ''): Variable
    {
        $new_var = new Variable(nd::arcsinh($this->getArray()));
        $new_var->registerOperation("arcsinh", [$this])->setName($name);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function arctanh(string $name = ''): Variable
    {
        $new_var = new Variable(nd::arctanh($this->getArray()));
        $new_var->registerOperation("arctanh", [$this])->setName($name);
        return $new_var;
    }
}