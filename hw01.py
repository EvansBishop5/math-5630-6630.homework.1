# Author: Your Name / your_email Evans Bishop/ Leb0134
# Date: 2024-09-01
# Assignment Name: hw01

import sys
import numpy as np

def p1():
    """
    This function only contains comments. Fill the following table. Do not write any code here.

    commands                                      |  results            | explanations
    ----------------------------------------------|---------------------|----------------
    import sys;sys.float_info.epsilon             | 2.22×10^−16          |I dont know how to code
    import sys;sys.float_info.max                 |1.797×10^308          | max number
    import sys;sys.float_info.min                 |2.225×10^−308         |min number
    import sys;1 + sys.float_info.epsilon - 1     | 1                    | not sure
    import sys;1 + sys.float_info.epsilon /2 - 1  |  5e-17                |small number
    import sys;sys.float_info.min/1e10            |2.2250738585072014e-30   |min/ by that value
    import sys;sys.float_info.min/1e16            |2.2250738585072014e-292 | min/ by that value
    import sys;sys.float_info.max*10              | infinity               | went over the max
    """

def p2(n, choice):
    """
    This function computes the Archimedes' method for pi.
    @param n: the number of sides of the polygon
    @param choice: 1 or 2, the formula to use
    @return: s_n, the approximation of pi using Archimedes' method.

    
    Tabulate the error of |s_n - pi| for n = 0, 1, 2, ... 15 and choices n = 1, 2
    for both choices of formulas.
    
    n     | choice 1 | choice 2
    ------|----------|----------
    0     |          |
    1     |          |
    2     |          |
    3     |          |
    4     |          |
    5     |          |
    6     |          |
    7     |          |
    8     |          |
    9     |          |
    10    |          |
    11    |          |
    12    |          |
    13    |          |
    14    |          |
    15    |          |
 

    Explanation of the results:


    """

    # Write your code here import numpy as np

def archimedes_formula_1(pn):
    return pn / np.sqrt(2 + np.sqrt(4 - pn**2))

def archimedes_formula_2(pn):
    return (2 * pn) / np.sqrt(4 - pn**2)

def compute_sn(pn, n):
    return (2**n) * 6 * pn

pi = np.pi
n_values = [1, 2, 3, 4, 5]  # Example values of n
results = []

for n in n_values:
    # Initial value
    pn = np.sqrt(1)

    # Using the first formula
    for _ in range(n):
        pn = archimedes_formula_1(pn)
    sn1 = compute_sn(pn, n)
    error1 = abs(sn1 - pi)

    # Reset pn
    pn = np.sqrt(1)

    # Using the second formula
    for _ in range(n):
        pn = archimedes_formula_2(pn)
    sn2 = compute_sn(pn, n)
    error2 = abs(sn2 - pi)

    results.append((n, error1, error2))

# Print results
for result in results:
    n, error1, error2 = result
    print(f"n={n}: Error (Formula 1) = {error1:.2e}, Error (Formula 2) = {error2:.2e}")

    if choice == 1:
        # Use the 1st formula 
        pass
    else:
        # Use the 2nd formula
        pass

def p3(a):
    """
    This function implements the Kahan summation algorithm. 

    @param a: a 1D numpy array of numbers
    @return: the Kahan sum of the array
    """
    return 0 # Write you code here. 
    def kahan_summation(x_list):
    # Number of elements in the list
    n = len(x_list)
    
    # Initialize variables
    j = 0
    e = 0.0  # Compensation error
    s = x_list[j]  # Initial sum
    
    while j < n - 1:
        j += 1
        x = x_list[j]
        y = x - e  # Remove the previous error
        t = s + y  # Perform the summation
        e = (t - s) - y  # Calculate the new error
        s = t  # Update the sum
    
    return s

# Example usage
numbers = [1.0, 1e-16, -1e-16, 1e-15]
result = kahan_summation(numbers)
print(f"Kahan summation result: {result}")

def p4(a):
    """
    This function tests the performance of Kahan summation algorithm 
    against naive summation algorithm.

    @param a: a 1D numpy array of numbers
    @return: no return

    @task: Test this function with a = np.random.rand(n) with various size n multiple times. Summarize your findings below.

    @findings: i dont really get what this is asking at all





    """
    single_a = a.astype(np.float32) # Convert the input array to single precision
    s = p3(a) # Kahan sum of double precision as the ground truth
    single_kahan_s = p3(single_a) # Kahan sum of single precision
    single_naive_s = sum(single_a) # Naive sum of single precision

    print(f"Error of Kahan sum under single precision: {s - single_kahan_s}")
    print(f"Error of Naive sum under single precision: {s - single_naive_s}")

def p5(a):
    """
    For 6630. 

    This function computes summation of a vector using pairwise summation.
    @param a: a vector of numbers
    @return: the summation of the vector a using pairwise summation algorithm.

    @note: You may need to create a helper function if your code uses recursion.

    @task: Rewrite the p4 test function to test this summation method. Summarize your findings below.
    
    @findings: 
    
    
    
    
    
    """

    return 0 # Write your code here.
import numpy as np

def recursive_summation(x_list, i, k):
    # Base case: single element
    if i == k:
        return x_list[i]
    
    # Find midpoint
    mid = (i + k) // 2
    
    # Recursive calls
    left_sum = recursive_summation(x_list, i, mid)
    right_sum = recursive_summation(x_list, mid + 1, k)
    
    # Return the combined sum
    return left_sum + right_sum

def compare_summations(numbers):
    # Convert to single precision
    numbers = np.array(numbers, dtype=np.float32)
    
    # Compute sums using both methods
    sum_kahan = kahan_summation(numbers)
    sum_recursive = recursive_summation(numbers, 0, len(numbers) - 1)
    
    # Compute true sum using double precision for comparison
    true_sum = np.float32(sum(numbers))
    
    # Calculate errors
    error_kahan = abs(np.float32(true_sum) - sum_kahan)
    error_recursive = abs(np.float32(true_sum) - sum_recursive)
    
    return sum_kahan, sum_recursive, error_kahan, error_recursive

# Example usage
numbers = [np.float32(1.0), np.float32(1e-7), np.float32(-1e-7), np.float32(1e-6)]

sum_kahan, sum_recursive, error_kahan, error_recursive = compare_summations(numbers)

print(f"Kahan summation result (single precision): {sum_kahan}")
print(f"Recursive summation result (single precision): {sum_recursive}")
print(f"Error in Kahan summation: {error_kahan:.10e}")
print(f"Error in Recursive summation: {error_recursive:.10e}")
