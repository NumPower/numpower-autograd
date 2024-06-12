<?php

namespace NumPower;
use ArithmeticOperand;
use Exception;
use NDArray as nd;
use NumPower\Core\Operand;

final class Tensor extends Operand
{
    /**
     * @throws Exception
     */
    public function __construct(int|float|array|object $value, string $name = "", bool $requireGrad = false)
    {
        if (is_int($value) || is_float($value) && $name == '') {
            $name = $value;
        }
        if (!is_a($value, '\NDArray')) {
            $value = nd::array($value);
        }
        if (is_a($value, '\NumPower\Core\Operand')) {

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
    public function getData(): \NDArray
    {
        return $this->getArray();
    }

    /**
     * @return string
     */
    public function __toString(): string
    {
        return $this->getArray();
    }

    /**
     * @param nd|float|Tensor $value
     * @return void
     */
    public function setData(\NDArray|float|Tensor $value): void
    {
        if (is_a($value, Tensor::class)) {
            $value = $value->getData();
        }
        $this->setArray($value);
    }
}