from itertools import accumulate


def generate_vibrant_red_blue_gradient(val):
    # val ranges from 0.0 to 1.0
    g = 0  # Keep green completely turned off to avoid yellows/greens

    if val <= 0.5:
        # First half (0.0 to 0.5): Red stays at max brightness, Blue ramps up
        # This smoothly transitions: Red -> Pink -> Vibrant Magenta/Purple
        r = 255
        b = int((val / 0.5) * 255)
    else:
        # Second half (0.5 to 1.0): Blue stays at max brightness, Red ramps down
        # This smoothly transitions: Vibrant Magenta/Purple -> Violet -> Blue
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