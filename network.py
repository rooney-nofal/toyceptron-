import math
from typing import List
from layer import Layer

class Network:
    """
    Represents a fully connected neural network (Multi-Layer Perceptron).
    """
    @staticmethod
    def sigmoid(x: float) -> float:
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    @staticmethod
    def relu(x: float) -> float:
        return max(0.0, x)

    @staticmethod
    def threshold(x: float) -> float:
        return 1.0 if x > 0 else 0.0
    
    @staticmethod
    def identity(x: float) -> float:
        return x

    def __init__(self, layer_sizes: List[int], activation: str = "sigmoid"):
        """
        Initialize the network architecture and activation function.
        
        Args:
            layer_sizes (List[int]): The architecture (e.g., [2, 3, 1]).
            activation (str): The activation function to use ('sigmoid', 'relu', 'threshold', 'identity').
        """
        self.layers: List[Layer] = []
        
        # 1. Select the activation function based on the string name
        if activation == "relu":
            self.activation_func = self.relu
        elif activation == "threshold":
            self.activation_func = self.threshold
        elif activation == "identity":
            self.activation_func = self.identity
        else:
            self.activation_func = self.sigmoid  # Default to Sigmoid

        # 2. Create the layers
        for i in range(len(layer_sizes) - 1):
            input_n = layer_sizes[i]
            output_n = layer_sizes[i+1]
            new_layer = Layer(input_size=input_n, layer_size=output_n)
            self.layers.append(new_layer)

    def forward(self, inputs: List[float]) -> List[float]:
        """
        Passes the input vector through the entire network, applying the activation function.
        """
        current_data = inputs
        
        for layer in self.layers:
            # Step A: Compute the weighted sum (Z) for the whole layer
            z_values = layer.forward(current_data)
            
            # Step B: Apply the activation function to each element
            # This is the "Spark" that makes the network non-linear
            current_data = [self.activation_func(z) for z in z_values]
            
        return current_data
    
    def summary(self):
        """
        Prints a summary of the network architecture.
        """
        print("-" * 30)
        print("Network Architecture Summary")
        print("-" * 30)
        
        for i, layer in enumerate(self.layers):
            # We look at the first neuron to determine input size
            n_inputs = len(layer.neurons[0].weights)
            n_neurons = len(layer.neurons)
            print(f"Layer {i+1}: Input {n_inputs} -> Output {n_neurons} Neurons")
            
        print("-" * 30)

if __name__ == "__main__":
    # --- Unit Test ---
    print("Running Network Unit Test...")
    
    # 1. Define Architecture: 2 inputs -> 3 hidden neurons -> 1 output neuron
    architecture = [2, 3, 1]
    my_network = Network(architecture)
    
    # 2. Test Summary
    my_network.summary()
    
    # 3. Test Forward Pass
    inputs = [1.0, 0.5]
    result = my_network.forward(inputs)
    
    print(f"Input: {inputs}")
    print(f"Output: {result}")
    
    # Check if output size matches the last layer size (1)
    assert len(result) == 1, f"Failed: Expected 1 output, got {len(result)}"
    print("Test Passed: Network logic is correct.")         