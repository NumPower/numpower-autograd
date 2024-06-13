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

- NumPower Extension >= 0.5.x (https://github.com/NumPower/numpower)
- PHP >= 8.3
- Composer

## Installing

```bash
$ composer require numpower/autograd
```
## Getting Started

The `NumPower\Tensor` class is a type of N-dimensional array with support for automatic differentiation. 
It uses the NDArray object from the NumPower extension as its engine, retaining many of the 
flexibilities of using an NDArray, such as using it as an operator in arimetic operations, 
simplifying code reading.

```php 
use NumPower\Tensor;

$a = new Tensor([[1, 2], [3, 4]], requireGrad: True);
$b = new Tensor([[5, 6], [7, 8]], requireGrad: True);

$c = (($a + $b) / $b)->sum();

$c->backward();

$dc_Da = $a->grad();
$dc_Db = $b->grad();

echo "out: \n";
echo $c;
echo "\ndc_Da: \n";
echo $dc_Da;
echo "dc_Db: \n";
echo $dc_Db;
```
```php
out: 
5.4619045257568
dc_Da: 
[[0.2, 0.166667]
 [0.142857, 0.125]]
dc_Db: 
[[-0.04, -0.0555556]
 [-0.0612245, -0.0625]]
```
This code demonstrates a simple example of automatic differentiation using variables and operations. It computes 
the gradients of the result of a computation (**$c**) with respect to its inputs (**$a** and **$b**).

## Using a video card with CUDA support
If you have compiled the NumPower extension with GPU utilization 
capabilities, you can perform operations on the GPU by allocating 
your `Tensor` in VRAM. The operations will automatically identify 
whether your `Tensor` is in RAM or VRAM and will perform the 
appropriate operation on the correct device.

```php
$a = new Tensor([[1, 2], [3, 4]], requireGrad: True, useGpu: True);
$b = new Tensor([[5, 6], [7, 8]], requireGrad: True, useGpu: True);

$c = ($a + $b)->sum(); // (Addition is performed on GPU)

$c->backward(); // Back propagation is performed on GPU
```
> `Tensor` works exactly the same as `NDArray` when dealing with different devices in multi-argument operations. Unless the argument is a scalar, all N-dimensional arguments of an operation must be stored on the same device.

## Simple training with autograd
Here we can see a more practical example. Let's create a neural network 
with a hidden layer and 1 output layer. Some common Neural Network 
functions are ready-made and can be accessed statically through 
the `NumPower\NeuralNetwork` module.



```php 
use NDArray as nd;
use NumPower\Tensor;
use NumPower\NeuralNetwork\Activations as activation;
use NumPower\NeuralNetwork\Losses as loss;

class SimpleModel
{
    public Tensor $weights_hidden_layer;
    public Tensor $weights_output_layer;
    public Tensor $hidden_bias;
    public Tensor $output_bias;
    private float $learningRate;

    public function __construct(int $inputDim = 2,
                                int $outputDim = 1,
                                int $hiddenSize = 16,
                                float $learningRate = 0.01
    )
    {
        $this->learningRate = $learningRate;
        // Initialize hidden layer weights
        $this->weights_hidden_layer = new Tensor(
            nd::uniform([$inputDim, $hiddenSize], -0.5, 0.5),
            name: 'weights_hidden_layer',
            requireGrad: True
        );
        // Initialize output layer weights
        $this->weights_output_layer = new Tensor(
            nd::uniform([$hiddenSize, $outputDim],-0.5, 0.5),
            name: 'weights_output_layer',
            requireGrad: True
        );
        // Initialize hidden layer bias
        $this->hidden_bias = new Tensor(
            nd::uniform([$hiddenSize],  -0.5, 0.5),
            name: 'hidden_bias',
            requireGrad: True
        );
        // Initialize output layer bias
        $this->output_bias = new Tensor(
            nd::uniform([$outputDim], -0.5, 0.5),
            name: 'output_bias',
            requireGrad: True
        );
    }

    public function forward(Tensor $x, Tensor $y): array
    {
        // Forward pass - Hidden Layer
        $x = $x->matmul($this->weights_hidden_layer) + $this->hidden_bias;
        $x = activation::ReLU($x); // ReLU Activation

        // Forward pass - Output Layer
        $x = $x->matmul($this->weights_output_layer) + $this->output_bias;
        $x = activation::sigmoid($x); // Sigmoid Activation

        // Binary Cross Entropy Loss
        $loss = loss::BinaryCrossEntropy($x, $y, name: 'loss');
        return [$x, $loss];
    }

    public function backward(Tensor $loss)
    {
        // Trigger autograd
        $loss->backward();

        // SGD (Optimizer) - Update Hidden Layer weights and bias
        $dw_dLoss = $this->weights_hidden_layer->grad();

        $this->weights_hidden_layer -= ($dw_dLoss * $this->learningRate);
        $this->weights_hidden_layer->resetGradients();

        $this->hidden_bias -= ($this->hidden_bias->grad() * $this->learningRate);
        $this->hidden_bias->resetGradients();

        // SGD (Optimizer) - Update Output Layer weights and bias
        $db_dLoss = $this->weights_output_layer->grad();

        $this->weights_output_layer -= ($db_dLoss * $this->learningRate);
        $this->weights_output_layer->resetGradients();

        $this->output_bias -= $this->output_bias - ($this->output_bias->grad() * $this->learningRate);
        $this->output_bias->resetGradients();
    }
}
```

> If you want to know more about each step of creating this model, access our documentation 
and see step-by-step instructions on how to build this model.


The above model can be used for several different classification problems. 
For simplicity, let's see if our model can solve the XOR problem.

```php 
$num_epochs = 4000;
$x = new Tensor(nd::array([[0, 0], [1, 0], [1, 1], [0, 1]]), name: 'x');
$y = new Tensor(nd::array([[0], [1], [0], [1]]), name: 'y');

$model = new SimpleModel();

$start = microtime(true);
for ($current_epoch = 0; $current_epoch < $num_epochs; $current_epoch++) {
    // Forward Pass
    [$prediction, $loss] = $model->forward($x, $y);
    // Backward Pass
    $model->backward($loss);
    echo "\n Epoch ($current_epoch): ".$loss->getArray();
}

echo "\nPredicted:\n";
print_r($model->forward($x, $y)[0]->toArray());
```
