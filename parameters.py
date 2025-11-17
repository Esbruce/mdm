import numpy as np

# half pipe geometry

half_pipe_radius = 10
half_pipe_length = 183
half_pipe_slope = np.radians(18)

# physical constants

g = 9.81
mu = 0.05 # coefficient of friction
C_d = 0.05 # drag coefficient
rho = 1.3 # air density
mass = 80.0 # mass of rider
A = 0.279 # frontal area of rider
epsilon = 1e-10 # avoid division by zero

# steering PD defaults (set to zero to preserve legacy behavior)
k_p = 30
k_d = 30
u_s_max = 300  # Increased to allow sufficient control force
# aliases expected by logic/simulate
k_p_default = k_p
k_d_default = k_d
u_s_max_default = u_s_max

# initial conditions for contact ODE
# Note: Use generate_initial_velocities() from functions.py in your script
# to calculate theta_dot_0 and s_dot_0 from speed
s_0 = 0.0
theta_0 = np.pi/2  # Start at right lip for S-curve swinging motion
s_dot_0 = 0
theta_dot_0 = 0

# Defining variables for air ODE

x_0 = 0.0
y_0 = 0.0
z_0 = 0.0
x_dot_0 = 0.0
y_dot_0 = 0.0
z_dot_0 = 0.0

# coordinate axis of local frame

e_x = np.array([1, 0, 0]) # same as world X axis (i) 
e_s = np.array([0, np.cos(half_pipe_slope), -np.sin(half_pipe_slope)])
# local normal axis: make the local frame right-handed with e_n = e_x × e_s
e_n = np.array([0, np.sin(half_pipe_slope), np.cos(half_pipe_slope)])

# coordinate axis of world frame

world_X = np.array([1,0,0])
world_Y = np.array([0, 1, 0])
world_Z = np.array([0, 0, 1])

max_segments = 10
max_airs = 6
use_steering=True

s_phase_max = 80
t_phase_max = 20
t_air_phase_max = 15

air_r_tol = 1e-6
air_a_tol = 1e-9

landing_tol = 1e-2