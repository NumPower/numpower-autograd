<?php

namespace NumPower\Tensor\Core\Math;

use Exception;
use NumPower\Tensor\Utils\ValidationUtils;
use NumPower\Tensor\Variable;
use NDArray as nd;

trait Arithmetics
{
    /**
     * @return \NDArray|float|int
     */
    abstract public function getArray(): \NDArray|float|int;

    /**
     * @throws Exception
     */
    public function add(int|float|array|object $value, string $name = ''): Variable
    {
        $input = ValidationUtils::validateOperationInputs($name, $value)[0];
        $output = new Variable(nd::add($this->getArray(), $input->getArray()));
        $output->registerOperation("add", [$this, $input]);
        $output->setName($name, $this);
        return $output;
    }

    /**
     * @param int|float|array|object $value
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function divide(int|float|array|object $value, string $name = ''): Variable
    {
        $input = ValidationUtils::validateOperationInputs($name, $value)[0];
        $output = new Variable(nd::divide($this->getArray(), $input->getArray()));
        $output->registerOperation("divide", [$this, $value])->setName($name, $this);
        return $output;
    }

    /**
     * @param int|float|array|object $value
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function multiply(int|float|array|object $value, string $name = ''): Variable
    {
        $input = ValidationUtils::validateOperationInputs($name, $value)[0];
        $output = new Variable(nd::multiply($this->getArray(), $input->getArray()));
        $output->registerOperation("multiply", [$this, $input])->setName($name, $this);
        return $output;
    }

    /**
     * @param int|float|array|object $value
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function power(int|float|array|object $value, string $name = ''): Variable
    {
        $input = ValidationUtils::validateOperationInputs($name, $value)[0];
        $new_var = new Variable($this->getArray() ** $input->getArray());
        $new_var->registerOperation("power", [$this, $input])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param int|float|array|object $y
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function mod(int|float|array|object $y, string $name = ''): Variable
    {
        $input = ValidationUtils::validateOperationInputs($name, $y)[0];
        $new_var = new Variable(nd::mod($this->getArray(), $input->getArray()));
        $new_var->registerOperation("mod", [$this, $input])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function negative(string $name = ''): Variable
    {
        $new_var = new Variable(nd::negative($this->getArray()));
        $new_var->registerOperation("negative", [$this])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param int|float|array|object $value
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function subtract(int|float|array|object $value, string $name = ''): Variable
    {
        $input = ValidationUtils::validateOperationInputs($name, $value)[0];
        $output = new Variable(nd::subtract($this->getArray(), $input->getArray()));
        $output->registerOperation("subtract", [$this, $input])->setName($name, $this);
        return $output;
    }
}