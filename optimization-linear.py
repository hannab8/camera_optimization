import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# LINEAR MODEL OF CALIBRATING THE CAMERA - DOES NOT TAKE DISTORTIONS INTO CONSIDERATION

'''
Observed points are the actual data points provided by the camera. This will be used
to run through the model.
'''
# Testing points to be replaced with real data provided by the camera
observed_points = np.array([
    [0, 3],
    [1, 5],
    [3.5, 10],
    [4.1, 11.2],
    [4.83, 12.66],
    [6.05, 15.1]
])

'''
This function represents a linear model being used. It takes parameters a = the slope and 
b = the y-intercept and calculates y for a given x based on the equation of a straight line (y = ax + b).
'''
def camera_model(params, x):
    a, b = params
    return a * x + b

'''
The objective_function finds the total squared error between what was observed
and the predicted values that were found through the model. This allows us to 
minimize the error.
'''
def objective_function(params, observed_points):
    total_error = 0
    for x, y in observed_points:
        predicted_y = camera_model(params, x)
        total_error += (predicted_y - y) ** 2
    return total_error

'''
The initial params are set as an initial 'guess', but will be adjusted until it finds
the minimum error.
'''
initial_params = [1, 0]
result = minimize(objective_function, initial_params, args=(observed_points,))
optimized_params = result.x
predicted_points = np.array([camera_model(optimized_params, x) for x, _ in observed_points])


'''
BELOW is for visual purposes only.
'''
# This will print the results
print("Optimized Parameters:")
print("Slope (a): {:.2f}, Intercept (b): {:.2f}\n".format(*optimized_params))

print("Comparison of Observed and Predicted Points:")
for (observed, predicted) in zip(observed_points, predicted_points):
    print("Observed: ({:.2f}, {:.2f}), Predicted: ({:.2f}, {:.2f})".format(observed[0], observed[1], observed[0], predicted))

# Plotting the observed and predicted points
plt.figure(figsize=(10, 6))
plt.scatter(*zip(*observed_points), color='blue', label='Observed Points')
plt.scatter(observed_points[:, 0], predicted_points, color='red', label='Predicted Points')

# Adding a line that represents the model
x_values = np.linspace(min(observed_points[:, 0]), max(observed_points[:, 0]), 100)
y_values = camera_model(optimized_params, x_values)
plt.plot(x_values, y_values, color='green', label='Fitted Line')

plt.title('Comparison of Observed and Predicted Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()



