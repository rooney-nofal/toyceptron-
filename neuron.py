import random
from typing import List, Optional

class Neuron: 
    """
    Represents a single unit of computation in a neural network layer.
    Performs a weighted sum of inputs plus a bias.
    """
    def __init__(self, input_size: int, weights: Optional[List[float]] = None, bias: Optional[float] = None):
        """
        Initialize the neuron.
        
        Args:
            input_size (int): The number of inputs this neuron receives.
            weights (Optional[List[float]]): Manually set weights. If None, randomized.
            bias (Optional[float]): Manually set bias. If None, randomized.
        """
        # Validation: If weights are provided, they must match the input size
        if weights is not None:
            if len(weights) != input_size:
                raise ValueError(f"Weight count ({len(weights)}) must match input size ({input_size}).")
            self.weights = weights
        else:
            # Random initialization between -1 and 1
            self.weights = [random.uniform(-1, 1) for _ in range(input_size)]

        # Initialize bias (randomly or fixed)
        self.bias = bias if bias is not None else random.uniform(-1, 1)

    def forward(self, inputs: List[float]) -> float:
        """
        Compute the output and SAVE it for backpropagation.
        """
        if len(inputs) != len(self.weights):
            raise ValueError(f"Input size ({len(inputs)}) must match weight size ({len(self.weights)}).")
        
        # SAVE the input (we need this to calculate the gradient later)
        self.last_input = inputs
        
        # Calculate dot product
        total = sum(i * w for i, w in zip(inputs, self.weights)) + self.bias
        
        # SAVE the pre-activation total (useful for some activation functions)
        self.last_total = total
        
        return total
    
    def update_weights(self, delta: float, learning_rate: float):
        """
        Update weights and bias using the error term (delta) and saved input.
        """
        # 1. Update Bias
        # CHANGE: Use += instead of -=
        # We ADD the correction because error = (Target - Output)
        self.bias += learning_rate * delta
        
        # 2. Update Weights
        for i in range(len(self.weights)):
            input_val = self.last_input[i]
            gradient = delta * input_val
            # CHANGE: Use += instead of -=
            self.weights[i] += learning_rate * gradient

if __name__ == "__main__":
    # --- Unit Test ---
    print("Running Neuron Unit Test...")
    
    # 1. Create a neuron with fixed weights (Acting like an AND gate)
    # Weights=[1, 1], Bias=-1.5
    # Logic: 1*1 + 1*1 - 1.5 = 0.5 (Positive)
    test_neuron = Neuron(input_size=2, weights=[1.0, 1.0], bias=-1.5)
    
    # 2. Run a forward pass with inputs [1, 1]
    inputs = [1.0, 1.0]
    result = test_neuron.forward(inputs)
    
    print(f"Inputs: {inputs}")
    print(f"Weights: {test_neuron.weights}, Bias: {test_neuron.bias}")
    print(f"Output: {result}")
    
    assert result == 0.5, f"Test Failed: Expected 0.5, got {result}"
    print("Test Passed: Neuron logic is correct.")
