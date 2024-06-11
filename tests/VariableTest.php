<?php
namespace NumPower\Tensor\Tests;

use Exception;
use NumPower\Tensor\Core\Operand;
use NumPower\Tensor\Tensor;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\TestCase;

#[CoversClass(Tensor::class)]
#[CoversClass(Operand::class)]
class VariableTest extends TestCase
{
    /**
     * @return void
     * @throws Exception
     */
    public function testVariableCreation()
    {
        $a = new Tensor([[1, 2, 3], [4, 5, 6]]);
        $this->assertIsArray($a->getValue());
        $this->assertArrayIsIdenticalToArrayIgnoringListOfKeys($a->getValue(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], []);
    }


    /**
     * @return void
     * @throws Exception
     */
    public function testVariableName()
    {
        $a = new Tensor([[1, 2, 3], [4, 5, 6]]);
        $a->setName('test');
        $this->assertEquals('test', $a->getName());
    }

    /**
     * @return void
     * @throws Exception
     */
    public function testOperandNumElements()
    {
        $a = new Tensor([[1, 2, 3], [4, 5, 6]]);
        $this->assertEquals(6, $a->numElements());
    }

    /**
     * @return void
     * @throws Exception
     */
    public function testVariableIndexing()
    {
        $a = new Tensor([[1, 2, 3], [4, 5, 6]]);

        $this->assertIsArray($a[0]->getValue());
        $this->assertArrayIsIdenticalToArrayIgnoringListOfKeys($a[0]->getValue(), [1.0, 2.0, 3.0], []);

        $this->assertIsFloat($a[0][1]->getValue());
        $this->assertEquals(2.0, $a[0][1]->getValue());
    }
}