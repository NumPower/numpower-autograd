<?php

namespace NumPower\Core\Math;

use Exception;
use NumPower\Utils\ValidationUtils;
use NumPower\Tensor;
use NDArray as nd;

trait Arithmetics
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
     * @throws Exception
     */
    public function add(int|float|array|object $value, string $name = ''): Tensor
    {
        $input = ValidationUtils::validateOperationInputs($name, $value)[0];
        $output = new Tensor(nd::add($this->getArray(), $input->getArray()), requireGrad: ($this->requireGrad() || $input->requireGrad()));
        $output->registerOperation("add", [$this, $input]);
        $output->setName($name, $this);
        return $output;
    }

    /**
     * @param int|float|array|object $value
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function divide(int|float|array|object $value, string $name = ''): Tensor
    {
        $input = ValidationUtils::validateOperationInputs($name, $value)[0];
        $output = new Tensor(nd::divide($this->getArray(), $input->getArray()), requireGrad: ($this->requireGrad() || $input->requireGrad()));
        $output->registerOperation("divide", [$this, $input])->setName($name, $this);
        return $output;
    }

    /**
     * @param int|float|array|object $value
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function multiply(int|float|array|object $value, string $name = ''): Tensor
    {
        $input = ValidationUtils::validateOperationInputs($name, $value)[0];
        $output = new Tensor(nd::multiply($this->getArray(), $input->getArray()), requireGrad: ($this->requireGrad() || $input->requireGrad()));
        $output->registerOperation("multiply", [$this, $input])->setName($name, $this);
        return $output;
    }

    /**
     * @param int|float|array|object $value
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function power(int|float|array|object $value, string $name = ''): Tensor
    {
        $input = ValidationUtils::validateOperationInputs($name, $value)[0];
        $new_var = new Tensor($this->getArray() ** $input->getArray(), requireGrad: ($this->requireGrad() || $input->requireGrad()));
        $new_var->registerOperation("power", [$this, $input])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param int|float|array|object $y
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function mod(int|float|array|object $y, string $name = ''): Tensor
    {
        $input = ValidationUtils::validateOperationInputs($name, $y)[0];
        $new_var = new Tensor(nd::mod($this->getArray(), $input->getArray()), requireGrad: ($this->requireGrad() || $input->requireGrad()));
        $new_var->registerOperation("mod", [$this, $input])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function negative(string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::negative($this->getArray()), requireGrad: ($this->requireGrad()));
        $new_var->registerOperation("negative", [$this])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param int|float|array|object $value
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function subtract(int|float|array|object $value, string $name = ''): Tensor
    {
        $input = ValidationUtils::validateOperationInputs($name, $value)[0];
        $output = new Tensor(nd::subtract($this->getArray(), $input->getArray()), requireGrad: ($this->requireGrad() || $input->requireGrad()));
        $output->registerOperation("subtract", [$this, $input])->setName($name, $this);
        return $output;
    }

    /**
     * @param bool $keepdim
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function sum(bool $keepdim = false, string $name = ''): Tensor
    {
        $value = nd::sum($this->getArray());
        if ($keepdim) {
            if (is_float($value)) {
                $value = nd::ones($this->getArray()->shape()) * $value;
            } elseif (count($value->shape()) == 1 && count($value) == 1) {
                $value = $value[0] * nd::ones($this->getArray()->shape());
            }
        }
        $new_var = new Tensor($value, requireGrad: $this->requireGrad());
        $new_var->registerOperation("sum", [$this, $keepdim])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param int $axis
     * @param bool $keepdim
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function sum_axis(int $axis, bool $keepdim = false, string $name = ''): Tensor
    {
        $value = nd::sum($this->getArray(), $axis);
        if ($keepdim) {
            if (count($value->shape()) == 1 && count($value) == 1) {
                $value = $value[0] * nd::ones($this->getArray()->shape());
            }
            if (count($value->shape()) == 1 && count($this->getArray()->shape()) == 2) {
                $value = nd::reshape($value, [count($value), 1]);
            }
        }
        $new_var = new Tensor($value, requireGrad: $this->requireGrad());
        $new_var->registerOperation("sum_axis", [$this, $axis, $keepdim])->setName($name, $this);
        return $new_var;
    }
}