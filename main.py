from network import Network

def main():
    print("========================================")
    print("   TOYCEPTRON v1.2 - SYSTEM CHECK       ")
    print("========================================")

    # --- Test Case 1: Standard Sigmoid Network ---
    print("\n[Test 1] Initializing Sigmoid Network...")
    # Architecture: 2 Inputs -> 3 Hidden Neurons -> 1 Output Neuron
    net_sigmoid = Network(layer_sizes=[2, 3, 1], activation="sigmoid")
    net_sigmoid.summary()
    
    input_vec = [0.5, -0.2]
    output = net_sigmoid.forward(input_vec)
    print(f"Input: {input_vec}")
    print(f"Output: {output}")
    
    # Verification: Sigmoid must be between 0 and 1
    assert 0.0 <= output[0] <= 1.0, "Error: Sigmoid output out of range!"
    print(">> Status: PASS (Sigmoid behavior confirmed)")

    # --- Test Case 2: ReLU Network (Deep Architecture) ---
    print("\n[Test 2] Initializing ReLU Network (Deep)...")
    # Architecture: 3 Inputs -> 4 Hidden -> 4 Hidden -> 2 Outputs
    net_relu = Network(layer_sizes=[3, 4, 4, 2], activation="relu")
    net_relu.summary()
    
    input_vec_2 = [1.0, -0.5, 0.0]
    output_relu = net_relu.forward(input_vec_2)
    print(f"Input: {input_vec_2}")
    print(f"Output: {output_relu}")
    
    # Verification: ReLU must be non-negative
    for val in output_relu:
        assert val >= 0.0, "Error: ReLU output cannot be negative!"
    print(">> Status: PASS (ReLU behavior confirmed)")

    print("\n========================================")
    print("       ALL SYSTEMS OPERATIONAL          ")
    print("========================================")

if __name__ == "__main__":
    main()