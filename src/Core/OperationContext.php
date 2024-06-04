<?php

namespace NumPower\Tensor\Core;

class OperationContext
{
    /**
     * @var string
     */
    private string $name;

    /**
     * @var callable
     */
    private $backfn;

    /**
     * @param string $name
     */
    public function __construct(string $name)
    {
        $this->setName($name);
    }

    /**
     * @param callable $back_fn
     * @return OperationContext
     */
    public function setBackwardFunction(callable $back_fn): OperationContext
    {
        $this->backfn = $back_fn;
        return $this;
    }

    /**
     * @param string $name
     * @return $this
     */
    public function setName(string $name): OperationContext
    {
        $this->name = $name;
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
     * @return callable
     */
    public function getBackwardFunction(): callable
    {
        return $this->backfn;
    }
}