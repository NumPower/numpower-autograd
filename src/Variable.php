<?php

namespace NumPower\Tensor;
use ArithmeticOperand;
use Exception;
use NDArray as nd;
use NumPower\Tensor\Core\Operand;
use NumPower\Tensor\Utils\ValidationUtils;

final class Variable extends Operand
{
    /**
     * @throws Exception
     */
    public function __construct(int|float|array|object $value, string $name = "", bool $requireGrad = true)
    {
        if (is_int($value) || is_float($value) && $name == '') {
            $name = $value;
        }
        if (!is_a($value, '\NDArray')) {
            $value = nd::array($value);
        }
        $this->setArray($value);
        $this->requireGrad = $requireGrad;
        $this->setName($name);
    }

    /**
     * Return a PHP value.
     *
     * @return array|float
     */
    public function getValue(): array|float
    {
        if (is_a($this->getArray(), \NDArray::class)) {
            return $this->getArray()->toArray();
        }
        return $this->getArray();
    }

    /**
     * @param float $min
     * @param float $max
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function clip(float $min, float $max, string $name = ''): Variable
    {
        [$input_min, $input_max] = ValidationUtils::validateOperationInputs($name, $min, $max);
        $output = new Variable(nd::clip($this->getArray(), $min, $max));
        $output->registerOperation("clip", [$this, $input_min, $input_max])->setName($name, $this);
        return $output;
    }



    public function trunc(string $name = ''): Variable
    {
        $new_var = new Variable(nd::trunc($this->getArray()));
        $new_var->registerOperation("trunc", [$this])->setName($name, $this);
        return $new_var;
    }

    public function floor(string $name = ''): Variable
    {
        $new_var = new Variable(nd::floor($this->getArray()));
        $new_var->registerOperation("floor", [$this])->setName($name, $this);
        return $new_var;
    }

    public function ceil(string $name = ''): Variable
    {
        $new_var = new Variable(nd::ceil($this->getArray()));
        $new_var->registerOperation("ceil", [$this])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param bool $keepdim
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function sum(bool $keepdim = false, string $name = ''): Variable
    {
        $value = nd::sum($this->getArray());
        if ($keepdim) {
            if (is_float($value)) {
                $value = nd::ones($this->getArray()->shape()) * $value;
            } elseif (count($value->shape()) == 1 && count($value) == 1) {
                $value = $value[0] * nd::ones($this->getArray()->shape());
            }
        }
        $new_var = new Variable($value);
        $new_var->registerOperation("sum", [$this, $keepdim])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param int $axis
     * @param bool $keepdim
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function sum_axis(int $axis, bool $keepdim = false, string $name = ''): Variable
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
        $new_var = new Variable($value);
        $new_var->registerOperation("sum_axis", [$this, $axis, $keepdim])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function rsqrt(string $name = ''): Variable
    {
        $new_var = new Variable(nd::rsqrt($this->getArray()));
        $new_var->registerOperation("rsqrt", [$this])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function mean(string $name = ''): Variable
    {
        $new_var = new Variable(nd::average($this->getArray()));
        $new_var->registerOperation("mean", [$this])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function abs(string $name = ''): Variable
    {
        $new_var = new Variable(nd::abs($this->getArray()));
        $new_var->registerOperation("abs", [$this])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function cond(string $name = ''): Variable
    {
        $new_var = new Variable(nd::cond($this->getArray()));
        $new_var->registerOperation("cond", [$this])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function svd(string $name = ''): Variable
    {
        $new_var = new Variable(nd::svd($this->getArray()));
        $new_var->registerOperation("svd", [$this])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function norm(string $name = ''): Variable
    {
        $new_var = new Variable(nd::norm($this->getArray()));
        $new_var->registerOperation("norm", [$this])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param int|float|array|object $y
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function arctan2(int|float|array|object $y, string $name = ''): Variable
    {
        $input = ValidationUtils::validateOperationInputs($name, $y)[0];
        $new_var = new Variable(nd::arctan2($this->getArray(), $input->getArray()));
        $new_var->registerOperation("arctan2", [$this, $input])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function sqrt(string $name = ''): Variable
    {
        $new_var = new Variable(nd::sqrt($this->getArray()));
        $new_var->registerOperation("sqrt", [$this])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function sinc(string $name = ''): Variable
    {
        $new_var = new Variable(nd::sinc($this->getArray()));
        $new_var->registerOperation("sinc", [$this])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param array $shape
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function reshape(array $shape, string $name = ''): Variable
    {
        $new_var = new Variable(nd::reshape($this->getArray(), $shape));
        $new_var->registerOperation("reshape", [$this, $shape])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Variable
     * @throws Exception
     */
    public function sigmoid(string $name = ''): Variable
    {
        $ones = new Variable(1);
        $output = $ones->divide($this->multiply(-1, name: $name)->exp(name: $name)->add($ones, name: $name), name: $name);
        $output->setName($name, $this);
        return $output;
    }

    /**
     * @return string
     */
    public function __toString(): string
    {
        return $this->getArray();
    }

    /**
     * @param nd|float $value
     * @return void
     */
    public function setValue(\NDArray|float $value): void
    {
        $this->setArray($value);
    }
}