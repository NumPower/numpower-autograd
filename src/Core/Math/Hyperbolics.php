<?php

namespace NumPower\Tensor\Core\Math;

use \NDArray as nd;
use Exception;
use NumPower\Tensor\Utils\ValidationUtils;
use NumPower\Tensor\Tensor;

trait Hyperbolics
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
    public function arccosh(string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::arccosh($this->getArray()));
        $new_var->registerOperation("arccosh", [$this])->setName($name);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function sinh(string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::sinh($this->getArray()));
        $new_var->registerOperation("sinh", [$this])->setName($name);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function tanh(string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::tanh($this->getArray()));
        $new_var->registerOperation("tanh", [$this])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function cosh(string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::cosh($this->getArray()));
        $new_var->registerOperation("cosh", [$this])->setName($name);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function arcsinh(string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::arcsinh($this->getArray()));
        $new_var->registerOperation("arcsinh", [$this])->setName($name);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function arctanh(string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::arctanh($this->getArray()));
        $new_var->registerOperation("arctanh", [$this])->setName($name);
        return $new_var;
    }

    /**
     * @param int|float|array|object $y
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function arctan2(int|float|array|object $y, string $name = ''): Tensor
    {
        $input = ValidationUtils::validateOperationInputs($name, $y)[0];
        $new_var = new Tensor(nd::arctan2($this->getArray(), $input->getArray()));
        $new_var->registerOperation("arctan2", [$this, $input])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function sinc(string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::sinc($this->getArray()));
        $new_var->registerOperation("sinc", [$this])->setName($name, $this);
        return $new_var;
    }
}