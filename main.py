import random
from network import Network

def main():
    print("========================================")
    print("      TOYCEPTRON - TRAINING MODE        ")
    print("========================================")
    
    # 1. Define the "XOR" Dataset
    # Inputs: [A, B] -> Target: [Result]
    training_data = [
        ([0.0, 0.0], [0.0]), # 0 XOR 0 = 0
        ([0.0, 1.0], [1.0]), # 0 XOR 1 = 1
        ([1.0, 0.0], [1.0]), # 1 XOR 0 = 1
        ([1.0, 1.0], [0.0])  # 1 XOR 1 = 0
    ]

    # 2. Initialize Network (TUNED FOR SUCCESS)
    # 2 Inputs -> 4 Hidden Neurons (More Brain Power) -> 1 Output
    print("Initializing Network...")
    net = Network(layer_sizes=[2, 4, 1], activation="sigmoid")
    net.summary()

    # 3. Training Loop (TUNED FOR SUCCESS)
    epochs = 20000      # Doubled the training time
    learning_rate = 1.0 # Doubled the learning speed
    
    print(f"\nTraining for {epochs} epochs...")
    
    for epoch in range(epochs):
        total_error = 0.0
        
        for inputs, target in training_data:
            net.train(inputs, target, learning_rate)
            
            # Check error
            prediction = net.forward(inputs)
            total_error += abs(target[0] - prediction[0])
            
        # Print progress every 2000 epochs
        if (epoch + 1) % 2000 == 0:
            print(f"Epoch {epoch+1}: Total Error = {total_error:.5f}")

    print("\nTraining Complete! Testing results:")
    print("-" * 30)

    # 4. Final Validation
    for inputs, target in training_data:
        prediction = net.forward(inputs)[0]
        rounded = 1 if prediction > 0.5 else 0
        
        print(f"Input: {inputs} -> Prediction: {prediction:.4f} -> Rounded: {rounded} (Target: {target[0]})")
        
        # We use a small range for error because floats are rarely exact
        assert rounded == target[0], "Failed to learn pattern!"

    print("-" * 30)
    print("SUCCESS: The Network has learned XOR!")

if __name__ == "__main__":
    main()