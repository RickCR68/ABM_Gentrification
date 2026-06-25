from itertools import accumulate
import math


def generate_vibrant_red_blue_gradient(income, min_val=0.1, max_val=1.1):
    # 1. Clamp income within the bounds and normalize to a 0.0 -> 1.0 range
    income = max(min_val, min(max_val, income))
    # log scale for color mapping: map income to a color gradient from red (low) to blue (high)
    # Normalize the income to a 0.0 -> 1.0 range
    val = (income - min_val) / (max_val - min_val)
    # Apply log scale for more visually distinct gradients
    val = math.log(val * (math.e - 1) + 1)

    g = 0  # No green

    # 2. Your vibrant split logic
    if val <= 0.5:
        # Red stays max, Blue ramps up (Red -> Pink -> Vibrant Magenta)
        r = 255
        b = int((val / 0.5) * 255)
    else:
        # Blue stays max, Red ramps down (Vibrant Magenta -> Violet -> Blue)
        r = int(((1.0 - val) / 0.5) * 255)
        b = 255

    return f"#{r:02x}{g:02x}{b:02x}"

def calculate_morans_i(values, weights):
    """
    Calculate Moran's I statistic for spatial autocorrelation.

    Parameters:
    - values: A 1D array of values (e.g., income levels).
    - weights: A 2D array of spatial weights (e.g., adjacency matrix).

    Returns:
    - Moran's I statistic.
    """
    n = len(values)
    mean_value = sum(values) / n
    numerator = 0.0
    denominator = 0.0

    for i in range(n):
        for j in range(n):
            numerator += weights[i][j] * (values[i] - mean_value) * (values[j] - mean_value)
        denominator += (values[i] - mean_value) ** 2

    morans_i = (n / sum(sum(weights))) * (numerator / denominator)
    return morans_i