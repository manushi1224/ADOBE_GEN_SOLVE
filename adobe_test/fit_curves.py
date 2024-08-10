import numpy as np


class FitCurves:
    MAXPOINTS = 10000

    @staticmethod
    def fit_curve(points, error):
        t_hat1 = FitCurves.compute_left_tangent(points, 0)
        t_hat2 = FitCurves.compute_right_tangent(points, len(points) - 1)
        result = []
        FitCurves.fit_cubic(points, 0, len(points) - 1, t_hat1, t_hat2, error, result)
        return result

    @staticmethod
    def fit_cubic(points, first, last, t_hat1, t_hat2, error, result):
        iteration_error = error * error
        n_pts = last - first + 1

        if n_pts == 2:
            dist = np.linalg.norm(points[first] - points[last]) / 3.0
            bez_curve = [
                points[first],
                points[first] + t_hat1 * dist,
                points[last] + t_hat2 * dist,
                points[last],
            ]
            result.extend(bez_curve[1:])
            return

        u = FitCurves.chord_length_parameterize(points, first, last)
        bez_curve = FitCurves.generate_bezier(points, first, last, u, t_hat1, t_hat2)
        max_error, split_point = FitCurves.compute_max_error(
            points, first, last, bez_curve, u
        )

        if max_error < error:
            result.extend(bez_curve[1:])
            return

        if max_error < iteration_error:
            for i in range(4):
                u_prime = FitCurves.reparameterize(points, first, last, u, bez_curve)
                bez_curve = FitCurves.generate_bezier(
                    points, first, last, u_prime, t_hat1, t_hat2
                )
                max_error, split_point = FitCurves.compute_max_error(
                    points, first, last, bez_curve, u_prime
                )
                if max_error < error:
                    result.extend(bez_curve[1:])
                    return
                u = u_prime

        t_hat_center = FitCurves.compute_center_tangent(points, split_point)
        FitCurves.fit_cubic(
            points, first, split_point, t_hat1, t_hat_center, error, result
        )
        FitCurves.fit_cubic(
            points, split_point, last, -t_hat_center, t_hat2, error, result
        )

    @staticmethod
    def generate_bezier(points, first, last, u_prime, t_hat1, t_hat2):
        n_pts = last - first + 1
        A = np.zeros((n_pts, 2, 2))
        C = np.zeros((2, 2))
        X = np.zeros(2)

        for i in range(n_pts):
            A[i, 0] = t_hat1 * FitCurves.B1(u_prime[i])
            A[i, 1] = t_hat2 * FitCurves.B2(u_prime[i])

        for i in range(n_pts):
            C[0, 0] += np.dot(A[i, 0], A[i, 0])
            C[0, 1] += np.dot(A[i, 0], A[i, 1])
            C[1, 1] += np.dot(A[i, 1], A[i, 1])
            tmp = points[first + i] - (
                points[first] * FitCurves.B0(u_prime[i])
                + points[first] * FitCurves.B1(u_prime[i])
                + points[last] * FitCurves.B2(u_prime[i])
                + points[last] * FitCurves.B3(u_prime[i])
            )
            X[0] += np.dot(A[i, 0], tmp)
            X[1] += np.dot(A[i, 1], tmp)

        C[1, 0] = C[0, 1]

        det_C0_C1 = C[0, 0] * C[1, 1] - C[1, 0] * C[0, 1]
        det_C0_X = C[0, 0] * X[1] - C[1, 0] * X[0]
        det_X_C1 = X[0] * C[1, 1] - X[1] * C[0, 1]

        alpha_l = 0.0 if det_C0_C1 == 0 else det_X_C1 / det_C0_C1
        alpha_r = 0.0 if det_C0_C1 == 0 else det_C0_X / det_C0_C1

        seg_length = np.linalg.norm(points[first] - points[last])
        epsilon = 1.0e-6 * seg_length

        if alpha_l < epsilon or alpha_r < epsilon:
            dist = seg_length / 3.0
            bez_curve = [
                points[first],
                points[first] + t_hat1 * dist,
                points[last] + t_hat2 * dist,
                points[last],
            ]
            return bez_curve

        bez_curve = [
            points[first],
            points[first] + t_hat1 * alpha_l,
            points[last] + t_hat2 * alpha_r,
            points[last],
        ]
        return bez_curve

    @staticmethod
    def reparameterize(points, first, last, u, bez_curve):
        n_pts = last - first + 1
        u_prime = np.zeros(n_pts)

        for i in range(n_pts):
            u_prime[i] = FitCurves.newton_raphson_root_find(
                bez_curve, points[first + i], u[i]
            )

        return u_prime

    @staticmethod
    def newton_raphson_root_find(Q, P, u):
        Q_u = FitCurves.bezier_ii(3, Q, u)
        Q1 = 3 * (Q[1:] - Q[:-1])
        Q2 = 2 * (Q1[1:] - Q1[:-1])
        Q1_u = FitCurves.bezier_ii(2, Q1, u)
        Q2_u = FitCurves.bezier_ii(1, Q2, u)

        numerator = np.dot(Q_u - P, Q1_u)
        denominator = np.dot(Q1_u, Q1_u) + np.dot(Q_u - P, Q2_u)

        if denominator == 0:
            return u

        return u - numerator / denominator

    @staticmethod
    def bezier_ii(degree, V, t):
        Vtemp = V.copy()
        for i in range(1, degree + 1):
            for j in range(degree - i + 1):
                Vtemp[j] = (1.0 - t) * Vtemp[j] + t * Vtemp[j + 1]
        return Vtemp[0]

    @staticmethod
    def B0(u):
        return (1.0 - u) ** 3

    @staticmethod
    def B1(u):
        return 3 * u * (1.0 - u) ** 2

    @staticmethod
    def B2(u):
        return 3 * u**2 * (1.0 - u)

    @staticmethod
    def B3(u):
        return u**3

    @staticmethod
    def compute_left_tangent(points, end):
        if end + 1 >= len(points):
            # Handle case where there is no next point
            raise ValueError("Not enough points to compute tangent.")
        t_hat1 = np.subtract(points[end + 1], points[end])
        t_hat1 /= np.linalg.norm(t_hat1)  # Normalize the tangent vector
        return t_hat1

    @staticmethod
    def compute_right_tangent(points, end):
        if end - 1 < 0:
            # Handle case where there is no previous point
            raise ValueError("Not enough points to compute tangent.")
        t_hat2 = np.subtract(points[end - 1], points[end])
        t_hat2 /= np.linalg.norm(t_hat2)  # Normalize the tangent vector
        return t_hat2

    @staticmethod
    def compute_center_tangent(points, center):
        V1 = points[center - 1] - points[center]
        V2 = points[center] - points[center + 1]
        t_hat_center = (V1 + V2) / 2.0
        return t_hat_center / np.linalg.norm(t_hat_center)

    @staticmethod
    def chord_length_parameterize(points, first, last):
        u = np.zeros(last - first + 1)
        u[0] = 0.0
        for i in range(first + 1, last + 1):
            u[i - first] = u[i - first - 1] + np.linalg.norm(points[i] - points[i - 1])
        u /= u[-1]
        return u

    @staticmethod
    def compute_max_error(points, first, last, bez_curve, u):
        max_dist = 0.0
        split_point = (last - first + 1) // 2

        for i in range(first + 1, last):
            P = FitCurves.bezier_ii(3, bez_curve, u[i - first])
            dist = np.linalg.norm(P - points[i]) ** 2
            if dist >= max_dist:
                max_dist = dist
                split_point = i

        return max_dist, split_point
points = np.vstack((x, y)).T

