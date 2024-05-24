<?php

namespace NumPower\Tensor\Core\Tape;

use Exception;
use NDArray as nd;
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
     * @param string $name
     * @param array $args
     */
    public function __construct(string $name, array $args)
    {
        $this->setName($name);
        $this->setArgs($args);
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
     * @param bool $benchmark
     * @return void
     * @throws Exception
     */
    public function backward(Variable $tensor, \NDArray|float|int $grad, bool $benchmark = False): void
    {
        if (!method_exists(BackwardOperation::class, $this->name)) {
            throw new Exception("Impossible to compute gradient of `$this->name`");
        }
        BackwardOperation::{$this->name}($tensor, $benchmark, $grad, ...$this->getArgs());
    }

    /**
     * @param Variable $tensor
     * @param bool $withHeader
     * @return void
     */
    public function backwardPrint(Variable $tensor, bool $withHeader = true): void
    {
        $names = [];
        $name = $tensor->getName();
        foreach ($this->getArgs() as $arg) {
            if (is_a($arg, Variable::class)) {
                $names[] = $arg->getName();
            } elseif (is_float($arg) || is_int($arg) || is_string($arg)) {
                $names[] = (string)$arg;
            }
        }
        $argsString = "[".implode(", ", $names)."]";
        $nameWidth = 20;
        $operationWidth = 20;
        $argsWidth = 40;
        if ($withHeader) {
            printf("%-{$nameWidth}s %-{$operationWidth}s %-{$argsWidth}s\n", "Tensor Name", "Operation", "Arguments");
            printf("%-{$nameWidth}s %-{$operationWidth}s %-{$argsWidth}s\n", str_repeat("=", $nameWidth), str_repeat("=", $operationWidth), str_repeat("=", $argsWidth));
        }

        printf("%-{$nameWidth}s %-{$operationWidth}s %-{$argsWidth}s\n", $name, $this->getName(), $argsString);
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
}