<?php

namespace NumPower\Core\Math;

use \NDArray as nd;
use Exception;
use NumPower\Utils\ValidationUtils;
use NumPower\Tensor;

trait Hyperbolics
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
    public function arccosh(string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::arccosh($this->getArray()), requireGrad: $this->requireGrad());
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
        $new_var = new Tensor(nd::sinh($this->getArray()), requireGrad: $this->requireGrad());
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
        $new_var = new Tensor(nd::tanh($this->getArray()), requireGrad: $this->requireGrad());
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
        $new_var = new Tensor(nd::cosh($this->getArray()), requireGrad: $this->requireGrad());
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
        $new_var = new Tensor(nd::arcsinh($this->getArray()), requireGrad: $this->requireGrad());
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
        $new_var = new Tensor(nd::arctanh($this->getArray()), requireGrad: $this->requireGrad());
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
        $new_var = new Tensor(nd::arctan2($this->getArray(), $input->getArray()), requireGrad: ($this->requireGrad() || $input->requireGrad()));
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
        $new_var = new Tensor(nd::sinc($this->getArray()), requireGrad: $this->requireGrad());
        $new_var->registerOperation("sinc", [$this])->setName($name, $this);
        return $new_var;
    }
}