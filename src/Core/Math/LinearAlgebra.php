<?php

namespace NumPower\Tensor\Core\Math;

use NDArray as nd;
use Exception;
use NumPower\Tensor\Utils\ValidationUtils;
use NumPower\Tensor\Variable;

trait LinearAlgebra
{
    /**
     * @return \NDArray|float|int
     */
    abstract public function getArray(): \NDArray|float|int;

    /**
     * @param int|float|array|object $value
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function matmul(int|float|array|object $value, string $name = ''): Variable
    {
        $input = ValidationUtils::validateOperationInputs($name, $value)[0];
        $output = new Variable(nd::matmul($this->getArray(), $input->getArray()));
        $output->registerOperation("matmul", [$this, $input])->setName($name, $this);
        return $output;
    }

    /**
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function matrix_rank(string $name = ''): Variable
    {
        $new_var = new Variable(nd::matrix_rank($this->getArray()));
        $new_var->registerOperation("matrix_rank", [$this])->setName($name);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function det(string $name = ''): Variable
    {
        $new_var = new Variable(nd::det($this->getArray()));
        $new_var->registerOperation("det", [$this])->setName($name);
        return $new_var;
    }

    /**
     * @param int|float|array|object $y
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function dot(int|float|array|object $y, string $name = ''): Variable
    {
        $input = ValidationUtils::validateOperationInputs($name, $y)[0];
        if (count($this->getShape()) != 1 || count($input->getShape()) != 1) {
            throw new Exception("dot operation can only compute the dot product of two 1D tensors.");
        }
        $new_var = new Variable(nd::dot($this->getArray(), $input->getArray()));
        $new_var->registerOperation("dot", [$this, $input]);
        $new_var->setName($name);
        return $new_var;
    }

    /**
     * @param int|float|array|object $vec2
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function outer(int|float|array|object $vec2, string $name = ''): Variable
    {
        $input = ValidationUtils::validateOperationInputs($name, $vec2)[0];
        $output = new Variable(nd::outer(nd::flatten($this->getArray()), nd::flatten($input->getArray())));
        $output->registerOperation("outer", [$this, $input]);
        return $output;
    }
}