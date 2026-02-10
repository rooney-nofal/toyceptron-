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
        Compute the output of the neuron for a given input vector.
        Formula: sum(input_i * weight_i) + bias
        
        Args:
            inputs (List[float]): The input vector.
            
        Returns:
            float: The weighted sum plus bias (pre-activation).
        """
        if len(inputs) != len(self.weights):
            raise ValueError(f"Input size ({len(inputs)}) must match weight size ({len(self.weights)}).")
            
        # Calculate dot product: sum(i * w)
        # zip(inputs, self.weights) pairs them up: [(i1, w1), (i2, w2)...]
        total = sum(i * w for i, w in zip(inputs, self.weights))
        
        # Add bias
        return total + self.bias

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
