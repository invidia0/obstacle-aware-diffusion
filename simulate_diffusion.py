import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.stats import multivariate_normal, gaussian_kde
from matplotlib.animation import FuncAnimation, PillowWriter

def animate_step(i, ax, particles, obstacle_points):
    ax.clear()
    ax.set_xlim(0, grid_size[0])
    ax.set_ylim(0, grid_size[1])
    ax.set_aspect('equal')
    ax.set_title(f"Step {i + 1} of {n_steps}")
    ax.set_xticks(np.arange(0, grid_size[0] + 1, 10))
    ax.set_yticks(np.arange(0, grid_size[1] + 1, 10))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.scatter(particles[:, 0], particles[:, 1], s=1, color='blue', alpha=0.5, label='Diffusing Particles')
    ax.scatter(obstacle_points[:, 0], obstacle_points[:, 1], color='red', s=10, label='Obstacle')
    ax.legend(loc='upper right')
    ax.grid(which='both', color='gray', linestyle='--', linewidth=0.5)
    print(f"Step {i + 1}: {len(particles)} free particles")
    plt.pause(0.1)

def plot_image(particles, obstacle_points, distribution, grid_size, xx, yy, block=False):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, grid_size[0])
    ax.set_ylim(0, grid_size[1])
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(0, grid_size[0] + 1, 10))
    ax.set_yticks(np.arange(0, grid_size[1] + 1, 10))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(ax.contourf(xx, yy, distribution, levels=20, cmap='viridis', alpha=1.0), label="Probability Density")
    ax.scatter(particles[:, 0], particles[:, 1], s=1, color='blue', alpha=0.5, label='Free')
    ax.scatter(obstacle_points[:, 0], obstacle_points[:, 1], color='red', s=10, label='Obstacle')
    ax.grid(which='both', color='gray', linestyle='--', linewidth=0.5)
    ax.legend(loc='upper right')
    plt.show(block=block)

# --- Parameters ---
n_particles = 10000
n_steps = 10 # Number of steps for diffusion, more steps = more spread
step_size = 1 # Travel distance per step for each particle
diffusion_coefficient = 1.0  # Diffusion coefficient
time_step = 1.0  # Time step for diffusion simulation

# --- Map parameters ---
grid_size = (100, 100) 
grid_resolution = 1.0  # meters per cell
obstacle_radius = 1.0  # collision threshold in map units
meshgrid_size = (int(grid_size[0] / grid_resolution), int(grid_size[1] / grid_resolution))
x = np.linspace(0, grid_size[0], meshgrid_size[0])
y = np.linspace(0, grid_size[1], meshgrid_size[1])
xx, yy = np.meshgrid(x, y)
grid_points = np.vstack([xx.ravel(), yy.ravel()]).T

# --- Define initial PDF (Gaussian centered in the wall) ---
mean = np.array([70, 50])
cov = np.array([[100, 0], [0, 100]])

# --- Generate a fake point cloud (wall) ---
wall_dim_x = 50
wall_dim_y = 10
wall_dim = (wall_dim_x, wall_dim_y)
wall_center = (grid_size[0] / 2, grid_size[1] / 2)
wall_x = np.linspace(wall_center[0] - wall_dim_x / 2, wall_center[0] + wall_dim_x / 2, wall_dim_x)
wall_y = np.linspace(wall_center[1] - wall_dim_y / 2, wall_center[1] + wall_dim_y / 2, wall_dim_y)

wall_points = np.array(np.meshgrid(wall_x, wall_y)).T.reshape(-1, 2)
obstacle_points = wall_points

# --- Create KDTree for obstacle points ---
obstacle_tree = KDTree(obstacle_points)

# Sample initial particles
initial_particles = np.random.multivariate_normal(mean, cov, size=n_particles)

# --- Simulate Brownian motion with obstacle rejection ---
particles = initial_particles.copy()
free_particles = np.empty((0, 2))

ground_truth_pdf = multivariate_normal.pdf(grid_points, mean=mean, cov=cov)

plot_image(initial_particles, obstacle_points, ground_truth_pdf.reshape(meshgrid_size), grid_size, xx, yy)

"""
-- GIF Animation --

fig, ax = plt.subplots(figsize=(6, 6))

def update(step):
    global particles
    ax.clear()

    proposals = particles + np.random.normal(scale=np.sqrt(2 * diffusion_coefficient * time_step), size=particles.shape)
    distances, _ = obstacle_tree.query(proposals, k=1)
    collision_mask = distances < obstacle_radius

    free = proposals[~collision_mask]
    colliding = proposals[collision_mask]

    # Plot particles
    ax.contourf(xx, yy, ground_truth_pdf.reshape(meshgrid_size), levels=20, cmap='Oranges', alpha=1)
    ax.scatter(free[:, 0], free[:, 1], s=1, color='blue', alpha=0.5, label='Free')
    ax.scatter(colliding[:, 0], colliding[:, 1], s=1, color='green', alpha=0.3, label='Collision')
    ax.scatter(wall_points[:, 0], wall_points[:, 1], color='red', s=10, label='Obstacle', alpha=0.5)

    ax.set_xlim(0, grid_size[0])
    ax.set_ylim(0, grid_size[1])
    ax.set_aspect('equal')
    ax.set_title(f"Step {step + 1}")
    ax.legend(loc='upper right')
    ax.set_xticks(np.arange(0, grid_size[0] + 1, 10))   
    ax.set_yticks(np.arange(0, grid_size[1] + 1, 10))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(which='both', color='gray', linestyle='--', linewidth=0.5)

    # Update only with free particles
    particles = free

# Create animation
anim = FuncAnimation(fig, update, frames=n_steps, interval=100)

# Save to GIF
anim.save("particle_diffusion_brownian.gif", writer=PillowWriter(fps=10))
plt.close()
"""

for t in range(n_steps):
    proposals = particles + np.random.normal(scale=step_size, size=particles.shape)
    # Check for collisions
    distances, _ = obstacle_tree.query(proposals, k=1)
    collision_mask = distances < obstacle_radius
    # Store the non-colliding particles
    free_particles = np.vstack((free_particles, proposals[~collision_mask]))
    # Update particles
    particles = proposals[~collision_mask]
    # Plot the current state
    animate_step(t, plt.gca(), particles, obstacle_points)


# --- Final PDF estimation ---
kde = gaussian_kde(particles.T, bw_method=0.15)
pdf_values = kde(grid_points.T).reshape(meshgrid_size)
pdf_values /= np.sum(pdf_values)  # Normalize

# --- Plot ---
plot_image(particles, obstacle_points, pdf_values, grid_size, xx, yy, block=True)
