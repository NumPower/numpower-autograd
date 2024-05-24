# NumPower Autograd

NumPower Autograd enriches the NumPower extension by enabling automatic 
differentiation through reverse-mode (backpropagation) automatic differentiation. Automatic 
differentiation is a computational technique used to efficiently and 
accurately compute derivatives of functions. Unlike numerical 
differentiation, which can be prone to errors and inefficiencies, 
automatic differentiation systematically applies the chain rule 
to compute gradients.

Reverse-mode automatic differentiation, commonly known as backpropagation, is particularly powerful in machine learning and optimization tasks. It calculates gradients of scalar functions with respect to N-dimensional arguments in a highly efficient manner, making it ideal for training complex models.

NumPower Autograd also supports GPU acceleration, leveraging the power of graphics processing units for faster computation. This capability enhances the library's performance, particularly for large-scale mathematical and machine learning tasks.

By integrating backpropagation and GPU support, NumPower Autograd provides high-performance gradient computation capabilities, facilitating advanced mathematical and machine learning applications.

---

## Requirements

- NumPower Extension > 0.5.x (https://github.com/NumPower/numpower)
- PHP 8.3
- Composer

## Installing

```bash
$ composer require numpower/autograd
```
## Getting Started

The `NumPower\Tensor\Variable` class is a type of N-dimensional array with support for automatic differentiation. 
It uses the NDArray object from the NumPower extension as its engine, retaining many of the 
flexibilities of using an NDArray, such as using it as an operator in arimetic operations, 
simplifying code reading.

```php 
use NumPower\Tensor\Variable;

$a = new Variable([[1, 2], [3, 4]]);
$b = new Variable([[5, 6], [7, 8]]);

$c = ($a + $b) / $b;

$c->backward();

$dc_Da = $a->grad();
$dc_Db = $b->grad();

echo "out: \n";
echo $c;
echo "dc_Da: \n";
echo $dc_Da;
echo "dc_Db: \n";
echo $dc_Db;
```
```php
out: 
[[1.2, 1.33333]
 [1.42857, 1.5]]
dc_Da: 
[[0.2, 0.166667]
 [0.142857, 0.125]]
dc_Db: 
[[-0.04, -0.0555556]
 [-0.0612245, -0.0625]]
```
This code demonstrates a simple example of automatic differentiation using variables and operations. It computes 
the gradients of the result of a computation (**$c**) with respect to its inputs (**$a** and **$b**).

## Creating custom operations
Sometimes you want to create specific operations that are not natively implemented in the library. In this case you can use the `operation` method to specify an operation and a backward propagation function for that operation.

``` php
// Custom operation function
function myop(OperationContext $context, ...$args): Variable
{
   $context->setName('myop');
   // Custom operation backward function
   $context->setBackwardFunction(
       function(Variable $output, \NDArray $grad, Variable $a, Variable $b) {
            $a->backward($grad * $a->getArray());
       }
   );
}

$a->operation(myop, $b);
```

## Using a video card with CUDA support
If you have compiled the NumPower extension with GPU utilization 
capabilities, you can perform operations on the GPU by allocating 
your Variable in VRAM. The operations will automatically identify 
whether your Variable is in RAM or VRAM and will perform the 
appropriate operation on the correct device.

```php
$a_gpu = $a->gpu();
$b_gpu = $b->gpu();

$c = $a + $b; // (Addition is performed on GPU)

$c->backward(); // Back propagation is performed on GPU

$a = $a_gpu->cpu(); // Copy data back to CPU
```
> `Variable` works exactly the same as `NDArray` when dealing with different devices in multi-argument operations. Unless the argument is a scalar, all N-dimensional arguments of an operation must be stored on the same device.

## Simple training with autograd
Here we can see a more practical example. Let's create a neural network 
with a hidden layer and 1 output layer. Some common Neural Network 
functions are ready-made and can be accessed statically through 
the `NumPower\Tensor\NeuralNetwork` module.



```php 
use NDArray as nd;
use NumPower\Tensor\Variable
use NumPower\Tensor\NeuralNetwork as nn;
use NumPower\Tensor\NeuralNetwork\Losses as loss;

class SimpleModel
{
    public Variable $weights_hidden_layer;
    public Variable $weights_output_layer;
    public Variable $hidden_bias;
    public Variable $output_bias;
    private float $learningRate;

    public function __construct(int $inputDim = 2,
                                int $outputDim = 1,
                                int $hiddenSize = 16,
                                float $learningRate = 0.001
    )
    {
        $this->learningRate = $learningRate;
        // Initialize hidden layer weights
        $this->weights_hidden_layer = new Variable(
            nd::uniform([$inputDim, $hiddenSize], -0.5, 0.5),
            name: 'weights_hidden_layer'
        );
        // Initialize output layer weights
        $this->weights_output_layer = new Variable(
            nd::uniform([$hiddenSize, $outputDim],-0.5, 0.5),
            name: 'weights_output_layer'
        );
        // Initialize hidden layer bias
        $this->hidden_bias = new Variable(
            nd::uniform([$hiddenSize],  -0.5, 0.5),
            name: 'hidden_bias'
        );
        // Initialize output layer bias
        $this->output_bias = new Variable(
            nd::uniform([$outputDim], -0.5, 0.5),
            name: 'output_bias'
        );
    }
    
    public function forward(Variable $x, Variable $y): array
    {
        // Forward pass - Hidden Layer
        $x = $x->matmul($this->weights_hidden_layer, name: 'hidden_matmul'); // Hidden Layer
        $x = $x->add($this->hidden_bias, name: 'add_bias_hidden'); // Add Bias
        $x = nn::SiLU($x, name: 'selu_activation'); // seLU Activation

        // Forward pass - Output Layer
        $x = $x->matmul($this->weights_output_layer, name: 'output_matmul');  // Output Layer
        $x = $x->add($this->output_bias, name: 'add_bias_output'); // Add Bias
        $x = $x->sigmoid(name: 'output_sigmoid'); // Sigmoid Activation

        // Mean Squared Error
        $loss = loss::MeanSquaredError($x, $y, name: 'loss');
        return [$x, $loss];
    }

    public function backward(Variable $loss)
    {
        // Trigger autograd
        $loss->backward();

        // SGD (Optimizer) - Update Hidden Layer weights and bias
        $dw_dLoss = $this->weights_hidden_layer->grad();

        $new_hidden_weights = $this->weights_hidden_layer->getArray() - ($dw_dLoss * $this->learningRate);
        $this->weights_hidden_layer->setArray($new_hidden_weights);

        $new_hidden_bias = $this->hidden_bias->getArray() - ($this->hidden_bias->grad() * $this->learningRate);
        $this->hidden_bias->setArray($new_hidden_bias);

        // SGD (Optimizer) - Update Output Layer weights and bias
        $db_dLoss = $this->weights_output_layer->grad();

        $new_output_weights = $this->weights_output_layer->getArray() - ($db_dLoss * $this->learningRate);
        $this->weights_output_layer->setArray($new_output_weights);

        $new_output_bias = $this->output_bias->getArray() - ($this->output_bias->grad() * $this->learningRate);
        $this->output_bias->setArray($new_output_bias);
    }
}
```

> If you want to know more about each step of creating this model, access our documentation 
and see step-by-step instructions on how to build this model.


The above model can be used for several different classification problems. 
For simplicity, let's see if our model can solve the XOR problem.