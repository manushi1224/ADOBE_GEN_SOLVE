import numpy as np
from fit_curves import FitCurves


# Sample list of arrays representing closed shapes

csv_path = "./isolated.csv"


def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=",")
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs


paths = read_csv("./frag0.csv")
tu = [tuple(p) for path in paths for p in path]
# print(tu)
# Define error tolerance
error_tolerance = 0.1

# # List to store fitted curves
fitted_curves = []

# for shape in paths:
#     # Convert the shape array into a list of tuples
#     points = [tuple(point) for point in shape]
#     print(points)

#     # Fit the curve to the points
bezier_curve = FitCurves.fit_curve(tu, error_tolerance)

#     # Append the fitted Bezier curve for this shape
fitted_curves.append(bezier_curve)

print(fitted_curves)
# fitted_curves now contains the Bezier curves for each closed shape
