import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def generate_src_points(n_points, field_length):
    px = (np.random.rand(n_points) - 0.5) * field_length
    py = (np.random.rand(n_points) - 0.5) * field_length
    return np.vstack((px, py)).T

def apply_transformation(points, transform):
    R = np.array([[np.cos(transform[2]), -np.sin(transform[2])],
                  [np.sin(transform[2]), np.cos(transform[2])]])
    return points @ R.T + transform[:2]

def add_noise(points, scale):
    return points + np.random.normal(scale=scale, size=points.shape)

def ndt_match(src_points, dst_points, resolution=7, max_iter=20, tol=1e-5):
    src_points = np.asarray(src_points)
    dst_points = np.asarray(dst_points)

    # Build a grid of cells with the specified resolution.
    min_corner = np.min(dst_points, axis=0) - resolution
    max_corner = np.max(dst_points, axis=0) + resolution
    grid_shape = np.ceil((max_corner - min_corner) / resolution).astype(int)
    grid = np.zeros(grid_shape, dtype=[('mean', float, 2), ('cov', float, (2, 2)), ('count', int)])

    # Fill the grid with destination points data.
    for p in dst_points:
        cell_idx = tuple(((p - min_corner) / resolution).astype(int))
        grid[cell_idx]['count'] += 1
        grid[cell_idx]['mean'] += (p - grid[cell_idx]['mean']) / grid[cell_idx]['count']
        grid[cell_idx]['cov'] += np.outer(p - grid[cell_idx]['mean'], p - grid[cell_idx]['mean'])

    # Calculate the covariance matrices for non-empty cells.
    for cell in grid.flat:
        if cell['count'] > 1:
            cell['cov'] /= cell['count'] - 1
        else:
            cell['cov'] = np.eye(2) * resolution

    # Optimization function to minimize.
    def objective_func(x):
        R = np.array([[np.cos(x[2]), -np.sin(x[2])], [np.sin(x[2]), np.cos(x[2])]])
        transformed_points = src_points @ R.T + x[:2]

        cost = 0.0
        for p in transformed_points:
            cell_idx = tuple(((p - min_corner) / resolution).astype(int))
            if 0 <= cell_idx[0] < grid_shape[0] and 0 <= cell_idx[1] < grid_shape[1]:
                cell = grid[cell_idx]
                if cell['count'] > 0:
                    d = p - cell['mean']
                    cost += d @ np.linalg.pinv(cell['cov']) @ d

        return cost

    def objective_func_vectorized(x):
        R = np.array([[np.cos(x[2]), -np.sin(x[2])], [np.sin(x[2]), np.cos(x[2])]])
        transformed_points = src_points @ R.T + x[:2]

        cell_idx = np.floor((transformed_points - min_corner) / resolution).astype(int)
        valid_mask = np.all(np.logical_and(cell_idx >= 0, cell_idx < grid_shape), axis=1)
        valid_cell_idx = cell_idx[valid_mask]

        valid_cells = grid[valid_cell_idx[:, 0], valid_cell_idx[:, 1]]
        count_mask = valid_cells['count'] > 0
        valid_cells = valid_cells[count_mask]
        valid_transformed_points = transformed_points[valid_mask][count_mask]

        d = valid_transformed_points - valid_cells['mean']
        cov_inv = np.linalg.pinv(valid_cells['cov'])
        cost = np.sum(np.einsum('...ij,...j->...i', d @ cov_inv, d))

        return cost


    # Optimize the transformation parameters.
    x0 = np.zeros(3)
    res = minimize(objective_func_vectorized, x0, method='BFGS', options={'maxiter': max_iter, 'gtol': tol})

    if not res.success:
        print("Warning: Optimization did not converge.")
        print("Message from optimizer:", res.message)
    else:
        print("Optimization successfully converged.")

    print("Final cost:", res.fun)
    print("Final parameters:", res.x)

    return res.x

def plot_results(src_points, dst_points, src_points_transformed):
    plt.scatter(src_points[:, 0], src_points[:, 1], c='r', label='Source Points')
    plt.scatter(dst_points[:, 0], dst_points[:, 1], c='b', label='Target Points')
    plt.scatter(src_points_transformed[:, 0], src_points_transformed[:, 1], c='g', label='Transformed Source Points')
    plt.legend()
    plt.axis('equal')
    plt.title('2D NDT Scan Matching')
    plt.show()

def main():
    n_points = 1000
    field_length = 50.0

    src_points = generate_src_points(n_points, field_length)
    true_transform = np.array([0.1, 0.2, np.pi / 12])
    dst_points = apply_transformation(src_points, true_transform)

    estimated_transform = ndt_match(src_points, dst_points)

    src_points_transformed = apply_transformation(src_points, estimated_transform)

    plot_results(src_points, dst_points, src_points_transformed)

    print(f"True transform: {true_transform}")
    print(f"Estimated transform: {estimated_transform}")

if __name__ == '__main__':
    main()