import random
from typing import List
from neuron import Neuron

class Layer:
    """
    Represents a layer of neurons in the network.
    A layer is a collection of neurons that all receive the same input.
    """
    def __init__(self, input_size: int, layer_size: int):
        """
        Initialize the layer.
        
        Args:
            input_size (int): The number of inputs each neuron receives.
            layer_size (int): The number of neurons in this layer.
        """
        self.neurons: List[Neuron] = []
        
        # Create 'layer_size' neurons, each expecting 'input_size' inputs
        for _ in range(layer_size):
            self.neurons.append(Neuron(input_size))

    def forward(self, inputs: List[float]) -> List[float]:
        """
        Passes the input vector through every neuron in the layer.
        
        Args:
            inputs (List[float]): The input vector.
            
        Returns:
            List[float]: The vector of outputs from all neurons in this layer.
        """
        output_vector = []
        
        # Each neuron processes the SAME input vector independently
        for neuron in self.neurons:
            neuron_output = neuron.forward(inputs)
            output_vector.append(neuron_output)
            
        return output_vector
    
    def backward(self, deltas: List[float], learning_rate: float) -> List[float]:
        """
        1. Calculate error to send back to previous layer.
        2. Update weights of all neurons in this layer.
        """
        input_size = len(self.neurons[0].weights)
        
        # Initialize gradient for previous layer with 0.0
        new_deltas = [0.0] * input_size
        
        # Loop through every neuron (j) and its corresponding error (delta)
        for j, neuron in enumerate(self.neurons):
            delta = deltas[j]
            
            # Accumulate gradient for previous layer: 
            # error_i = sum( weight_ji * delta_j )
            # We do this BEFORE updating weights to preserve the gradient logic
            for i in range(input_size):
                new_deltas[i] += neuron.weights[i] * delta
            
            # Now update the neuron (weights & bias)
            neuron.update_weights(delta, learning_rate)
            
        return new_deltas
    
if __name__ == "__main__":
    # --- Unit Test ---
    print("Running Layer Unit Test...")
    
    # 1. Create a layer with 3 neurons, expecting 2 inputs each
    input_size = 2
    layer_size = 3
    test_layer = Layer(input_size, layer_size)
    
    # 2. Define an input vector
    inputs = [1.0, 2.0]
    
    # 3. Run forward pass
    outputs = test_layer.forward(inputs)
    
    print(f"Input: {inputs}")
    print(f"Layer Size: {layer_size} Neurons")
    print(f"Output Vector: {outputs}")
    
    # Check if we got the right number of outputs
    assert len(outputs) == layer_size, f"Failed: Expected {layer_size} outputs, got {len(outputs)}"
    print("Test Passed: Layer logic is correct.")