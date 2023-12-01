import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# NONLINEAR MODEL of calibrating the camera that takes distortions into consideration

'''
Observed points are the actual data points provided by the camera. This will be used
to run through the model.
'''
# Testing points (to be replaced with real data from the camera)
observed_points = np.array([
    [0, 3],
    [1, 5],
    [3.5, 10],
    [4.1, 11.2],
    [4.83, 12.66],
    [6.05, 15.1]
])

'''
The model is designed to take the linear transformation into account, but it also includes
radial and tangential distortion.
'''
def nonlinear_camera_model(params, x, y):
    a, b, k1, k2, p1, p2 = params

    # Linear transformation
    x_lin = a * x + b
    y_lin = a * y + b

    # Radial distortion
    r_squared = x_lin**2 + y_lin**2
    radial_distortion = 1 + k1 * r_squared + k2 * r_squared**2

    x_radial = x_lin * radial_distortion
    y_radial = y_lin * radial_distortion

    # Tangential distortion
    x_tangential = x_radial + 2 * p1 * x_lin * y_lin + p2 * (r_squared + 2 * x_lin**2)
    y_tangential = y_radial + p1 * (r_squared + 2 * y_lin**2) + 2 * p2 * x_lin * y_lin

    return x_tangential, y_tangential

'''
The objective_function finds the total squared error between what was observed
and the predicted values that were found through the model. This allows us to 
minimize the error.
'''
def objective_function(params, observed_points):
    total_error = 0
    for (x_obs, y_obs) in observed_points:
        x_pred, y_pred = nonlinear_camera_model(params, x_obs, y_obs)
        total_error += (x_pred - x_obs) ** 2 + (y_pred - y_obs) ** 2
    return total_error

'''
The initial params are set as an initial 'guess', but will be adjusted until it finds
the minimum error. It was suggested that the parameters for the distortion 
coefficients are set to 0.
'''
initial_params = [1, 0, 0, 0, 0, 0]  # a, b, k1, k2, p1, p2
result = minimize(objective_function, initial_params, args=(observed_points,))
optimized_params = result.x
predicted_points = np.array([nonlinear_camera_model(optimized_params, x, y) for x, y in observed_points])



'''
BELOW is for visual purposes only.
'''
print("Optimized Parameters:")
print("a: {:.2f}, b: {:.2f}, k1: {:.2f}, k2: {:.2f}, p1: {:.2f}, p2: {:.2f}".format(*optimized_params))

print("\nComparison of Observed and Predicted Points:")
for (observed, predicted) in zip(observed_points, predicted_points):
    print(f"Observed: ({observed[0]:.2f}, {observed[1]:.2f}), Predicted: ({predicted[0]:.2f}, {predicted[1]:.2f})")

# Plotting the observed and predicted points
plt.figure(figsize=(10, 6))
plt.scatter(observed_points[:, 0], observed_points[:, 1], color='blue', label='Observed Points')
plt.scatter(*zip(*predicted_points), color='red', label='Predicted Points')

plt.title('Comparison of Observed and Predicted Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
