<?php

namespace NumPower\Autograd\Tests;

use Exception;
use NumPower\Tensor\Core\Math\Arithmetics;
use NumPower\Tensor\Tensor;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\UsesClass;
use PHPUnit\Framework\TestCase;

#[CoversClass(Arithmetics::class)]
#[CoversClass(Tensor::class)]
#[UsesClass(Tensor::class)]
class ArithmeticTest extends TestCase
{
    /**
     * @var Tensor
     */
    private Tensor $a;

    /**
     * @param string $name
     * @throws Exception
     */
    public function __construct(string $name)
    {
        $this->a = new Tensor([[1, 2, 3], [4, 5, 6]]);
        parent::__construct($name);
    }

    /**
     * @return void
     */
    public function testAdd()
    {
        $result_scalar_add = $this->a + 1;
        $this->assertArrayIsIdenticalToArrayIgnoringListOfKeys($result_scalar_add->getValue(), [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], []);
    }

    /**
     * @return void
     */
    public function testMultiply()
    {
        $result_scalar_add = $this->a * 2;
        $this->assertArrayIsIdenticalToArrayIgnoringListOfKeys($result_scalar_add->getValue(), [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]], []);
    }

    /**
     * @return void
     */
    public function testDivide()
    {
        $result_scalar_add = $this->a / 2;
        $this->assertArrayIsIdenticalToArrayIgnoringListOfKeys($result_scalar_add->getValue(), [[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]], []);
    }

    /**
     * @return void
     */
    public function testSubtract()
    {
        $result_scalar_add = $this->a - 2;
        $this->assertArrayIsIdenticalToArrayIgnoringListOfKeys($result_scalar_add->getValue(), [[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]], []);
    }

    /**
     * @return void
     */
    public function negative()
    {
        $result_scalar_add = -$this->a;
        $this->assertArrayIsIdenticalToArrayIgnoringListOfKeys($result_scalar_add->getValue(), [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], []);
    }
}