import numpy as np

def is_point_inside_ellipse(x, y, ellipse_params):
    x0, y0, a, b, theta_deg = ellipse_params
    # Convert angle to radians and invert it due to inverted Y-axis
    theta = np.deg2rad(theta_deg)
    
    # Translate point to ellipse center
    x_rel = x - x0
    y_rel = y - y0

    # Rotate point to align with ellipse axes
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x_prime = x_rel * cos_theta + y_rel * sin_theta
    y_prime = -x_rel * sin_theta + y_rel * cos_theta

    s = (x_prime / a) ** 2 + (y_prime / b) ** 2
    return s <= 1

def compute_min_distance(ellipse1, ellipse2, N=360, debug=False):
    # Unpack parameters from the pandas Series
    x0_1, y0_1 = ellipse1['x'], ellipse1['y']
    a1_full, b1_full = ellipse1['major'], ellipse1['minor']
    theta1_deg = ellipse1['angle']
    
    x0_2, y0_2 = ellipse2['x'], ellipse2['y']
    a2_full, b2_full = ellipse2['major'], ellipse2['minor']
    theta2_deg = ellipse2['angle']
    
    # Convert axes lengths to semi-major and semi-minor axes
    a1, b1 = a1_full / 2, b1_full / 2
    a2, b2 = a2_full / 2, b2_full / 2
    
    # Convert angles from degrees to radians
    theta1 = np.deg2rad(theta1_deg)
    theta2 = np.deg2rad(theta2_deg)

    t_values = np.linspace(0, 2 * np.pi, N, endpoint=False)
    cos_t = np.cos(t_values)
    sin_t = np.sin(t_values)

    # Compute points on the perimeter of the first ellipse
    ellipse1_x = x0_1 + a1 * cos_t * np.cos(theta1) - b1 * sin_t * np.sin(theta1)
    ellipse1_y = y0_1 + a1 * cos_t * np.sin(theta1) + b1 * sin_t * np.cos(theta1)

    # Compute points on the perimeter of the second ellipse
    ellipse2_x = x0_2 + a2 * cos_t * np.cos(theta2) - b2 * sin_t * np.sin(theta2)
    ellipse2_y = y0_2 + a2 * cos_t * np.sin(theta2) + b2 * sin_t * np.cos(theta2)
    if debug:
        return ellipse1_x, ellipse1_y, ellipse2_x, ellipse2_y

    # Check for overlap: First ellipse points inside the second ellipse
    for x, y in zip(ellipse1_x, ellipse1_y):
        if is_point_inside_ellipse(x, y, (x0_2, y0_2, a2, b2, theta2_deg)):
            return -1

    # Check for overlap: Second ellipse points inside the first ellipse
    for x, y in zip(ellipse2_x, ellipse2_y):
        if is_point_inside_ellipse(x, y, (x0_1, y0_1, a1, b1, theta1_deg)):
            return -1

    # Compute the minimum distance between the perimeters
    D_min = float('inf')
    for x1, y1 in zip(ellipse1_x, ellipse1_y):
        distances = np.hypot(ellipse2_x - x1, ellipse2_y - y1)
        min_distance = np.min(distances)
        if min_distance < D_min:
            D_min = min_distance

    return D_min
