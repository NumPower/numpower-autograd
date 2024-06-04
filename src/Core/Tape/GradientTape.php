<?php

namespace NumPower\Tensor\Core\Tape;

use Exception;
use NDArray as nd;
use NumPower\Tensor\Core\OperationContext;
use NumPower\Tensor\Variable;

class GradientTape
{
    /**
     * @var string
     */
    private string $name;

    /**
     * @var array
     */
    private array $args;

    /**
     * @var OperationContext|null
     */
    private ?OperationContext $context = null;

    /**
     * @param string $name
     * @param array $args
     * @param OperationContext|null $context
     */
    public function __construct(string $name, array $args, ?OperationContext $context)
    {
        $this->setName($name);
        $this->setArgs($args);
        $this->setContext($context);
    }

    /**
     * @param string $name
     * @return void
     */
    public function setName(string $name): void
    {
        $this->name = $name;
    }

    /**
     * @param array $args
     * @return void
     */
    public function setArgs(array $args): void
    {
        $this->args = $args;
    }

    /**
     * @return string
     */
    public function getName(): string
    {
        return $this->name;
    }

    /**
     * @return array
     */
    public function getArgs(): array
    {
        return $this->args;
    }

    /**
     * @param Variable $tensor
     * @param nd|float|int $grad
     * @return void
     * @throws Exception
     */
    public function diff(Variable $tensor, \NDArray|float|int $grad): void
    {
        if (isset($this->context)) {
            $this->getContext()->getBackwardFunction()($tensor, $grad, ...$this->getArgs());
            return;
        }
        if (!method_exists(BackwardOperation::class, $this->name)) {
            throw new Exception("Impossible to compute gradient of `$this->name`");
        }
        BackwardOperation::{$this->name}($tensor, $grad, ...$this->getArgs());
    }

    /**
     * @param Variable $tensor
     * @param bool $withHeader
     * @return void
     */
    public function backwardPrint(Variable $tensor, bool $withHeader = true): void
    {
        $names = [];
        foreach ($this->getArgs() as $arg) {
            if (is_a($arg, Variable::class)) {
                $name = $arg->getName();
                if ($name == '') {
                    $name = '_nd_';
                }
                $names[] = $name;
            } elseif (is_float($arg) || is_int($arg) || is_string($arg)) {
                $names[] = $arg;
            }
        }
        $argsString = "[".implode(", ", $names)."]";
        $operationWidth = 20;
        $argsWidth = 40;
        if ($withHeader) {
            printf("%-{$operationWidth}s %-{$argsWidth}s\n", "Operation", "Arguments");
            printf("%-{$operationWidth}s %-{$argsWidth}s\n", str_repeat("=", $operationWidth), str_repeat("=", $argsWidth));
        }

        printf("%-{$operationWidth}s %-{$argsWidth}s\n", $this->getName(), $argsString);
        foreach ($this->getArgs() as $arg) {
            if (is_a($arg, Variable::class)) {
                $tape = $arg->getTape();
                if ($tape == null) {
                    continue;
                }
                $tape->backwardPrint($arg, false);
            }
        }
    }

    /**
     * @param OperationContext|null $context
     * @return void
     */
    private function setContext(?OperationContext $context)
    {
        $this->context = $context;
    }

    /**
     * @return OperationContext
     */
    private function getContext(): OperationContext
    {
        return $this->context;
    }
}