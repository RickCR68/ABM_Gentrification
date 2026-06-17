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