<?php

namespace NumPower\Core\Math;

use NDArray as nd;
use Exception;
use NumPower\Utils\ValidationUtils;
use NumPower\Tensor;

trait LinearAlgebra
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
     * @param int|float|array|object $value
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function matmul(int|float|array|object $value, string $name = ''): Tensor
    {
        $input = ValidationUtils::validateOperationInputs($name, $value)[0];
        $output = new Tensor(nd::matmul($this->getArray(), $input->getArray()), requireGrad:($this->requireGrad() || $input->requireGrad()));
        $output->registerOperation("matmul", [$this, $input])->setName($name, $this);
        return $output;
    }

    /**
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function matrix_rank(string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::matrix_rank($this->getArray()), requireGrad: $this->requireGrad());
        $new_var->registerOperation("matrix_rank", [$this])->setName($name);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function det(string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::det($this->getArray()), requireGrad: $this->requireGrad());
        $new_var->registerOperation("det", [$this])->setName($name);
        return $new_var;
    }

    /**
     * @param int|float|array|object $y
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function dot(int|float|array|object $y, string $name = ''): Tensor
    {
        $input = ValidationUtils::validateOperationInputs($name, $y)[0];
        if (count($this->getShape()) != 1 || count($input->getShape()) != 1) {
            throw new Exception("dot operation can only compute the dot product of two 1D tensors.");
        }
        $new_var = new Tensor(nd::dot($this->getArray(), $input->getArray()), requireGrad: ($this->requireGrad() || $input->requireGrad()));
        $new_var->registerOperation("dot", [$this, $input]);
        $new_var->setName($name);
        return $new_var;
    }

    /**
     * @param int|float|array|object $vec2
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function outer(int|float|array|object $vec2, string $name = ''): Tensor
    {
        $input = ValidationUtils::validateOperationInputs($name, $vec2)[0];
        $output = new Tensor(nd::outer(nd::flatten($this->getArray()), nd::flatten($input->getArray())), requireGrad: ($this->requireGrad() || $input->requireGrad()));
        $output->registerOperation("outer", [$this, $input]);
        return $output;
    }

    /**
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function cond(string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::cond($this->getArray()), requireGrad: $this->requireGrad());
        $new_var->registerOperation("cond", [$this])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function svd(string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::svd($this->getArray()), requireGrad: $this->requireGrad());
        $new_var->registerOperation("svd", [$this])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function norm(string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::norm($this->getArray()), requireGrad: $this->requireGrad());
        $new_var->registerOperation("norm", [$this])->setName($name, $this);
        return $new_var;
    }
}