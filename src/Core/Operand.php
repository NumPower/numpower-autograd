<?php

namespace NumPower\Tensor\Core;

use ArithmeticOperand;
use Exception;
use NDArray as nd;
use NumPower\Tensor\Core\Math\Arithmetics;
use NumPower\Tensor\Core\Math\ExponentsLog;
use NumPower\Tensor\Core\Math\Hyperbolics;
use NumPower\Tensor\Core\Math\LinearAlgebra;
use NumPower\Tensor\Core\Math\Trigonometrics;
use NumPower\Tensor\Core\Tape\GradientTape;
use NumPower\Tensor\Variable;

abstract class Operand extends ArithmeticOperand
{
    use Arithmetics,
        ExponentsLog,
        Hyperbolics,
        Trigonometrics,
        LinearAlgebra;

    /**
     * @var mixed
     */
    private string $name = "";

    /**
     * @var \NDArray
     */
    protected \NDArray|float $array;

    /**
     * @var nd
     */
    protected \NDArray|float $grad;

    /**
     * @var GradientTape
     */
    protected GradientTape $tape;

    /**
     * @var bool
     */
    protected bool $requireGrad = true;

    /**
     * @param int|float|array|object $b
     * @return Variable
     * @throws Exception
     */
    public function __add(int|float|array|object $b) {
        return $this->add($b);
    }

    /**
     * @param int|float|array|object $b
     * @return Variable
     * @throws Exception
     */
    public function __mul(int|float|array|object $b) {
        return $this->multiply($b);
    }

    /**
     * @param int|float|array|object $b
     * @return Variable
     * @throws Exception
     */
    public function __pow(int|float|array|object $b) {
        return $this->power($b);
    }

    /**
     * @param int|float|array|object $b
     * @return Variable
     * @throws Exception
     */
    public function __div(int|float|array|object $b) {
        return $this->divide($b);
    }

    /**
     * @param int|float|array|object $b
     * @return Variable
     * @throws Exception
     */
    public function __sub(int|float|array|object $b) {
        return $this->subtract($b);
    }

    /**
     * @param int|float|array|object $b
     * @return Variable
     * @throws Exception
     */
    public function __mod(int|float|array|object $b) {
        return $this->mod($b);
    }

    /**
     * @param string $name
     * @param array $args
     * @return $this
     */
    public function registerOperation(string $name, array $args): Variable
    {
        if (!isset($this->tape)) {
            $this->tape = new GradientTape($name, $args);
        }
        return $this;
    }

    /**
     * @return string
     */
    public function getName(): string
    {
        return $this->name;
    }

    /**
     * @return GradientTape|null
     */
    public function getTape(): ?GradientTape
    {
        if (!isset($this->tape)) {
            return null;
        }
        return $this->tape;
    }

    /**
     * @return nd|float|int
     */
    public function getArray(): \NDArray|float|int
    {
        return $this->array;
    }

    /**
     * @return nd
     * @throws Exception
     */
    public function grad(): \NDArray|float
    {
        if (!isset($this->grad)){
            throw new Exception("No gradient found for `$this->name`.");
        }
        return $this->grad;
    }

    /**
     * @return array
     */
    public function getShape(): array
    {
        return $this->getArray()->shape();
    }

    /**
     * @param \NDArray $array
     * @return $this
     */
    public function setArray(\NDArray $array): Variable
    {
        $this->array = $array;
        return $this;
    }

    /**
     * @param nd|float|int|null $grad
     * @return void
     * @throws Exception
     */
    public function backward(\NDArray|float|int $grad = null, $benchmark = False)
    {
        if (!isset($grad)) {
            if (!is_float($this->getArray()) && !is_int($this->getArray())) {
                $grad = nd::ones($this->getArray()->shape());
            } else {
                $grad = nd::array(1);
            }
        }
        if ($this->requireGrad) {
            if (!isset($this->grad)) {
                $this->grad = $grad;
            } else {
                $this->grad += $grad;
            }
            $this->getTape()?->backward($this, $grad, $benchmark);
        }
    }

    /**
     * @throws Exception
     */
    public function graph(): void
    {
        if (!isset($this->tape)) {
            throw new Exception("The variable has no computable gradients");
        }
        $this->getTape()->backwardPrint($this);
    }

    /**
     * @param string $name
     * @return $this
     */
    public function setName(string $name): Variable
    {
        $this->name = $name;
        return $this;
    }
}