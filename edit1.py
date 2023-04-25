
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# from scipy.integrate import ode
import sys
import time

# Define the geometry and fluid properties
num_lines = 5
line_spacing = 1.0  # meters
fluid_density = 1000.0  # kg/m^3 (water)
gravity = 9.81  # m/s^2
incident_angle = np.pi / 4  # 45 degrees
nu = 1e-6  # m^2/s (kinematic viscosity of water)

# Define the computational domain
domain_size = (10, 10)  # meters
grid_resolution = 100

# Set up the grid
x = np.linspace(0, domain_size[0], grid_resolution)
y = np.linspace(0, domain_size[1], grid_resolution)
X, Y = np.meshgrid(x, y)
dx, dy = x[1] - x[0], y[1] - y[0]

# Define the initial conditions (e.g., velocity and pressure fields)
U0 = np.zeros((grid_resolution, grid_resolution, 2))
P0 = np.zeros((grid_resolution, grid_resolution))

# Define the conservation equations
def mass_conservation(U, P, dx, dy):
    dudx, dvdy = np.gradient(U[..., 0], dx), np.gradient(U[..., 1], dy)
    div_U = dudx + dvdy
    return div_U

def momentum_conservation(U, P, fluid_density, gravity, dx, dy, nu):
    dudx, dudy = np.gradient(U[..., 0], dx, axis=0), np.gradient(U[..., 0], dy, axis=1)
    dvdx, dvdy = np.gradient(U[..., 1], dx, axis=0), np.gradient(U[..., 1], dy, axis=1)
    dpdx, dpdy = np.gradient(P, dx, axis=0), np.gradient(P, dy, axis=1)
    
    d2udx2, d2udy2 = np.gradient(dudx, dx, axis=0), np.gradient(dudy, dy, axis=1)
    d2vdx2, d2vdy2 = np.gradient(dvdx, dx, axis=0), np.gradient(dvdy, dy, axis=1)
    
    u_dudx = U[..., 0] * dudx
    v_dudy = U[..., 1] * dudy
    u_dvdx = U[..., 0] * dvdx
    v_dvdy = U[..., 1] * dvdy
    
    viscous_x = nu * (d2udx2 + d2udy2)
    viscous_y = nu * (d2vdx2 + d2vdy2)

    pressure_x = -1 * (1 / fluid_density) * dpdx
    pressure_y = -1 * (1 / fluid_density) * dpdy
    
    U_new = np.zeros_like(U)
    U_new[..., 0] = U[..., 0] + u_dudx + v_dudy + pressure_x + viscous_x
    U_new[..., 1] = U[..., 1] + u_dvdx + v_dvdy + pressure_y + viscous_y - gravity
    
    return U_new

# Define the boundary conditions
def boundary_conditions(U, P):
    # Set velocities to zero at the domain boundaries
    U[:, 0, :] = 0
    U[:, -1, :] = 0
    U[0, :, :] = 0
    U[-1, :, :] = 0

    # Set velocities to zero at the vertical lines
    for i in range(num_lines):
        line_x_index = np.argmin(np.abs(x - (i+1)*line_spacing))
        U[:, line_x_index, :] = 0

# Define the main solver function
def solve_fluid_flow(t, y):
    U = np.reshape(y[:2*grid_resolution**2], (grid_resolution, grid_resolution, 2))
    P = np.reshape(y[2*grid_resolution**2:], (grid_resolution, grid_resolution))
    
    div_U = mass_conservation(U, P, dx, dy)
    U_new = momentum_conservation(U, P, fluid_density, gravity, dx, dy, nu)
    boundary_conditions(U_new, P)
    
    dydt = np.concatenate((U_new.flatten(), P.flatten()))
    return dydt

# Solve the system of equations
y0 = np.concatenate((U0.flatten(), P0.flatten()))

class ProgressBar:
    def __init__(self, total_iterations):
        self.total_iterations = total_iterations
        self.current_iteration = 0

    def update(self):
        self.current_iteration += 1
        progress = self.current_iteration / self.total_iterations * 100
        sys.stdout.write(f"\rProgress: {progress:.2f}%")
        sys.stdout.flush()

# Update the solver function to include the progress bar
def solve_fluid_flow_with_progress(t, y):
    progress_bar.update()
    return solve_fluid_flow(t, y)

# Set up the progress bar
progress_bar = ProgressBar(total_iterations=10000)

# Solve the system of equations using the updated solver function
sol = solve_ivp(solve_fluid_flow_with_progress, (0, 10), y0, method='BDF', t_eval=np.linspace(0, 10, 10000))


# Extract the steady-state solution
U_steady = np.reshape(sol.y[:2*grid_resolution**2, -1], (grid_resolution, grid_resolution, 2))
P_steady = np.reshape(sol.y[2*grid_resolution**2:, -1], (grid_resolution, grid_resolution))

# Plot the streamlines
plt.figure(figsize=(10, 10))
plt.streamplot(X, Y, U_steady[..., 0], U_steady[..., 1], density=2, linewidth=1, color='k')
plt.title('Streamlines at Steady State')
plt.xlabel('x [m]')
plt.ylabel('y [m]')

# Plot the array of vertical lines
for i in range(num_lines):
    plt.axvline(x=(i+1)*line_spacing, ymin=0, ymax=domain_size[1], color='r', linewidth=2)

plt.show()
