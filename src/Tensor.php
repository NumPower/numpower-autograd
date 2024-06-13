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
    public function __construct(int|float|array|object $value,
                                string $name = "",
                                bool $requireGrad = false,
                                bool $useGpu = false
    )
    {
        if (is_int($value) || is_float($value) && $name == '') {
            $name = $value;
        }
        if (!is_a($value, '\NDArray')) {
            $value = nd::array($value);
        }
        if ($useGpu && (is_a($value, '\NDArray') && !$value->isGPU())) {
            $value = $value->gpu();
        }
        $this->setArray($value);
        $this->requireGrad = $requireGrad;
        $this->setName($name);
    }

    /**
     * @return nd|float
     */
    public function getData(): \NDArray|float
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

    /**
     * @return bool
     */
    public function isGPU(): bool
    {
        $value = $this->getArray();
        if (is_scalar($value)) {
            return false;
        }
        return $value->isGPU();
    }
}