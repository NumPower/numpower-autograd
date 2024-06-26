<?php

namespace NumPower\Core;

use ArithmeticOperand;
use ArrayAccess;
use Exception;
use NDArray as nd;
use NumPower\Core\Math\Arithmetics;
use NumPower\Core\Math\ExponentsLog;
use NumPower\Core\Math\Hyperbolics;
use NumPower\Core\Math\LinearAlgebra;
use NumPower\Core\Math\Mathematical;
use NumPower\Core\Math\Rounding;
use NumPower\Core\Math\Trigonometrics;
use NumPower\Core\Tape\GradientTape;
use NumPower\Tensor;

abstract class Operand extends ArithmeticOperand implements ArrayAccess
{
    use Mathematical,
        Arithmetics,
        ExponentsLog,
        Hyperbolics,
        Trigonometrics,
        LinearAlgebra,
        Rounding;

    /**
     * @var mixed
     */
    private string $name = "";

    /**
     * @var nd|float
     */
    protected nd|float $array;

    /**
     * @var nd|float|null
     */
    protected nd|float|null $grad;

    /**
     * @var GradientTape
     */
    protected ?GradientTape $tape;

    /**
     * @var bool
     */
    protected bool $requireGrad = true;

    /**
     * @var ?Operand
     */
    protected ?Operand $origin = null;

    /**
     * @param int|float|array|object $b
     * @return Tensor
     * @throws Exception
     */
    public function __add(int|float|array|object $b) {
        return $this->add($b);
    }

    /**
     * @param int|float|array|object $b
     * @return Tensor
     * @throws Exception
     */
    public function __mul(int|float|array|object $b) {
        return $this->multiply($b);
    }

    /**
     * @param int|float|array|object $b
     * @return Tensor
     * @throws Exception
     */
    public function __pow(int|float|array|object $b) {
        return $this->power($b);
    }

    /**
     * @param int|float|array|object $b
     * @return Tensor
     * @throws Exception
     */
    public function __div(int|float|array|object $b) {
        return $this->divide($b);
    }

    /**
     * @param int|float|array|object $b
     * @return Tensor
     * @throws Exception
     */
    public function __sub(int|float|array|object $b) {
        return $this->subtract($b);
    }

    /**
     * @param int|float|array|object $b
     * @return Tensor
     * @throws Exception
     */
    public function __mod(int|float|array|object $b) {
        return $this->mod($b);
    }

    /**
     * @param string $name
     * @param array $args
     * @param OperationContext|null $context
     * @return $this
     */
    public function registerOperation(string $name, array $args, ?OperationContext $context = null): Tensor
    {
        if (!isset($this->tape)) {
            $this->tape = new GradientTape($name, $args, $context);
        }
        return $this;
    }

    public function resetGradients(): void
    {
        $this->grad = null;
        $this->tape = null;
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
     * Return the NDArray or scalar value
     *
     * @return nd|float|int
     */
    public function getArray(): \NDArray|float|int
    {
        return $this->array;
    }

    /**
     * This attribute is NULL by default and becomes a Tensor the first time a call to backward()
     * computes gradients for self. The attribute will then contain the gradients computed and
     * future calls to backward() will accumulate (add) gradients into it.
     *
     * @return Tensor
     * @throws Exception
     */
    public function grad(): Tensor
    {
        if (!isset($this->grad)){
            throw new Exception("No gradient found for `$this->name`.");
        }
        return new Tensor($this->grad);
    }

    /**
     * Returns the shape of the Tensor or 0 if the Tensor is a scalar.
     *
     * @return array|int
     */
    public function getShape(): array|int
    {
        if (is_scalar($this->getArray())) {
            return 0;
        }
        return $this->getArray()->shape();
    }

    /**
     * @param nd|float|int $array
     * @return $this
     */
    protected function setArray(\NDArray|float|int $array): Tensor
    {
        $this->array = $array;
        return $this;
    }

    /**
     * @return bool
     */
    public function requireGrad(): bool
    {
        return $this->requireGrad;
    }

    /**
     * @param nd|float|int|null $grad
     * @return void
     * @throws Exception
     */
    public function diff(\NDArray|float|int $grad = null): void
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
            $this->getTape()?->diff($this, $grad);
        }
    }

    /**
     * Returns true if the tensor is a scalar or false if it is an n-dimensional array.
     *
     * @return bool
     */
    public function isScalar(): bool
    {
        return is_scalar($this->getArray());
    }

    /**
     * Computes the gradient of current tensor graph leaves.
     *
     * @param nd|float|int|null $grad
     * @return void
     * @throws Exception
     */
    public function backward(\NDArray|float|int $grad = null): void
    {
        if (!$this->isScalar()) {
            throw new Exception("grad can only be created for scalar outputs");
        }
        $this->diff($grad);
    }

    /**
     * Print the Tensor graph
     *
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
     * @param Tensor|null $origin
     * @return $this
     */
    public function setName(string $name, ?Tensor $origin = null): Tensor
    {
        if ($name == '' && $origin != null) {
            $name = $origin->getName();
        }
        $this->name = $name;
        return $this;
    }

    /**
     * @param callable $operation
     * @param ...$args
     * @return Tensor
     * @throws Exception
     */
    public function operation(callable $operation, ...$args): Tensor
    {
        $context = new OperationContext('custom_operation');
        $forward_args = [];
        foreach ($args as $idx => $arg) {
            if (is_a($arg, Tensor::class)) {
                $forward_args[] = $arg->getArray();
                continue;
            }
            $forward_args[] = $arg;
        }
        // @var Tensor $result
        $result = $operation($context, $this->getArray(), ...$forward_args);
        if (!is_a($result, Tensor::class) && !is_a($result, \NDArray::class) && !is_scalar($result)) {
            throw new Exception("Invalid return for operation `".$context->getName()."`.");
        }
        if (is_a($result, \NDArray::class) || is_scalar($result)) {
            $result = new Tensor($result);
        }
        $result->registerOperation($context->getName(), array_merge([$this], $args), $context)->setName('out_'.$context->getName());
        return $result;
    }

    /**
     * @param mixed $offset
     * @return Tensor
     * @throws Exception
     */
    public function offsetGet(mixed $offset): mixed
    {
        $view = $this->getArray()[$offset];
        $output = new Tensor($view);
        $output->registerOperation('offsetGet', [$this, $offset])->setName('out_'.$offset.'_offset', $this);
        return $output;
    }

    /**
     * @param mixed $offset
     * @param mixed $value
     * @return void
     */
    public function offsetSet(mixed $offset, mixed $value): void
    {
        $this->getArray()[$offset] = $value;
    }

    /**
     * @param mixed $offset
     * @return void
     */
    public function offsetUnset(mixed $offset): void
    {
        $this->getArray()[$offset] = NAN;
    }

    /**
     * @param mixed $offset
     * @return bool
     */
    public function offsetExists(mixed $offset): bool
    {
        return (count($this->getArray()) > $offset);
    }

    /**
     * Returns the total number of elements in the tensor.
     *
     * @return int
     */
    public function numElements(): int
    {
        if (is_scalar($this->getArray())) {
            return 1;
        }
        return $this->getArray()->size();
    }

    /**
     * Gives a new shape to an array without changing its data.
     *
     * @param array $shape
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function reshape(array $shape, string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::reshape($this->getArray(), $shape), requireGrad: $this->requireGrad());
        $new_var->registerOperation("reshape", [$this, $shape])->setName($name, $this);
        return $new_var;
    }

    /**
     * Converts the buffer data to a PHP object, array or scalar.
     *
     * @return array|float
     */
    public function toArray(): array|float
    {
        return $this->getArray()->toArray();
    }

    /**
     * Returns a new Tensor, detached from the current graph.
     * The result will never require gradient.
     *
     * @return Tensor
     * @throws Exception
     */
    public function detach(): Tensor
    {
        return new Tensor($this->array);
    }
}