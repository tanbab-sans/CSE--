import numpy as np

A = np.random.randint(1, 10, size=(4, 4))
B = np.random.randint(1, 10, size=(4, 4))

C = A @ B

try:
    C_inv = np.linalg.inv(C)
except np.linalg.LinAlgError:

    print("Inverse matrix does not exist. Please run again.")
    exit()

I = C @ C_inv

print("--- Matrix C ---")
print(C)
print("\n--- Inverse Matrix C ---")
print(C_inv)
print("\n--- Identity Matrix (C * C_inv) ---")
print(np.round(I, 2)) 