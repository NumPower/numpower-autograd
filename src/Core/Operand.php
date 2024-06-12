<?php

namespace NumPower\Core;

use \ArithmeticOperand;
use ArrayAccess;
use Exception;
use NDArray as nd;
use NumPower\Core\Math\Arithmetics;
use NumPower\Core\Math\ExponentsLog;
use NumPower\Core\Math\Hyperbolics;
use NumPower\Core\Math\LinearAlgebra;
use NumPower\Core\Math\Rounding;
use NumPower\Core\Math\Trigonometrics;
use NumPower\Core\Tape\GradientTape;
use NumPower\Tensor;

abstract class Operand extends ArithmeticOperand implements ArrayAccess
{
    use Arithmetics,
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
    public function diff(\NDArray|float|int $grad = null)
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
     * @return bool
     */
    public function isScalar()
    {
        return is_float($this->array);
    }

    /**
     * @param nd|float|int|null $grad
     * @return void
     * @throws Exception
     */
    public function backward(\NDArray|float|int $grad = null, $benchmark = False)
    {
        if ($this->isScalar() == false) {
            throw new Exception("grad can be created only for scalar outputs");
        }
        return $this->diff($grad);
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
        if (!is_a($result, Tensor::class) && !is_a($result, \NDArray::class)) {
            throw new Exception("Invalid return for operation `".$context->getName()."`.");
        }
        if (is_a($result, \NDArray::class)) {
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
     * @param array $shape
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function reshape(array $shape, string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::reshape($this->getArray(), $shape));
        $new_var->registerOperation("reshape", [$this, $shape])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function rsqrt(string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::rsqrt($this->getArray()));
        $new_var->registerOperation("rsqrt", [$this])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function mean(string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::average($this->getArray()), requireGrad: $this->requireGrad());
        $new_var->registerOperation("mean", [$this])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function abs(string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::abs($this->getArray()));
        $new_var->registerOperation("abs", [$this])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function sqrt(string $name = ''): Tensor
    {
        $new_var = new Tensor(nd::sqrt($this->getArray()));
        $new_var->registerOperation("sqrt", [$this])->setName($name, $this);
        return $new_var;
    }

    /**
     * @param string $name
     * @return Tensor
     * @throws Exception
     */
    public function sigmoid(string $name = ''): Tensor
    {
        $ones = new Tensor(1, requireGrad: $this->requireGrad());
        $output = $ones->divide($this->multiply(-1, name: $name)->exp(name: $name)->add($ones, name: $name), name: $name);
        $output->setName($name, $this);
        return $output;
    }
}