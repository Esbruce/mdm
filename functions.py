import numpy as np
import math
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

''' Simulation Parameters '''

# half pipe geometry

half_pipe_radius = 3.0
half_pipe_length = 100.0
half_pipe_slope = 0.0

# physical constants

g = 9.81
mu = 0.05 # coefficient of friction
C_d = 0.85 # drag coefficient
rho = 1.3 # air density
mass = 80.0 # mass of rider
A = 0.279 # frontal area of rider
epsilon = 1e-10 # avoid division by zero

# initial conditions for contact ODE

s_0 = 0.0
theta_0 = np.pi/2 - 0.05
s_dot_0 = 10
theta_dot_0 = -2

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

''' Functions returning the ODEs for the contact and air phases'''

def contact_ODE_system(t, y, half_pipe_radius, half_pipe_length, half_pipe_slope, g, mu, mass):
    # y = [theta, s, theta_dot, s_dot]
    theta, s, theta_dot, s_dot = y
    
    # avoid division by zero
    speed = math.sqrt(s_dot**2 + half_pipe_radius**2 * theta_dot**2)
    if speed < epsilon:
        speed = epsilon

    # ODE for s
    s_ddot = np.sin(half_pipe_slope) * g - mu * (half_pipe_radius * theta_dot ** 2 - g * np.cos(half_pipe_slope) * np.cos(theta)) * (s_dot / speed)

    # ODE for theta
    theta_ddot = - ((g * np.cos(half_pipe_slope) * np.sin(theta))/ half_pipe_radius) - mu * (half_pipe_radius * theta_dot ** 2 + g * np.cos(half_pipe_slope) * np.cos(theta)) * (theta_dot / speed)

    # Return first-order derivatives: [theta_dot, s_dot, theta_ddot, s_ddot]
    return [theta_dot, s_dot, theta_ddot, s_ddot]

def air_ODE_system(t, y, g, mass, C_d, A, rho):
    # y = [x, y, z, x_dot, y_dot, z_dot]
    x, y, z, x_dot, y_dot, z_dot = y
    
    # Velocity magnitude
    speed = math.sqrt(x_dot*x_dot + y_dot*y_dot + z_dot*z_dot)

    # Drag coefficient factor
    k = 0.5 * rho * C_d * A / mass

    # Drag acceleration components (Z is vertical)
    x_ddot = -k * speed * x_dot
    y_ddot = -k * speed * y_dot
    z_ddot = -g - k * speed * z_dot

    # Return first-order derivatives: [x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot]
    return [x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot]

'''Functions for solving the ODEs with event detection incorporated'''

def solve_contact_ODE(half_pipe_radius, half_pipe_length, half_pipe_slope, g, mu, mass, s_0, theta_0, s_dot_0, theta_dot_0, stop_at="pos"):

    # Initial conditions

    theta = theta_0
    s = s_0
    theta_dot = theta_dot_0
    s_dot = s_dot_0
    y_0 = [theta, s, theta_dot, s_dot]

    # Time span
    t_span = (0, 1000)

    # Events detection: takeoff when reaching either lip |theta| = pi/2
    def lip_event(t, y, *args):
        theta = y[0]
        return abs(theta) - (math.pi / 2)

    lip_event.terminal = True
    lip_event.direction = 1  # crossing from inside toward the lip

    # Along-pipe limit: stop if |s| exceeds half the pipe length
    def s_limit_event(t, y, *args):
        s = y[1]
        return abs(s) - (half_pipe_length / 2.0)

    s_limit_event.terminal = True
    s_limit_event.direction = 0

    # Solve the ODE

    sol = solve_ivp(
        contact_ODE_system,
        t_span,
        y_0,
        args=(half_pipe_radius, half_pipe_length, half_pipe_slope, g, mu, mass),
        events=[lip_event, s_limit_event],
        rtol=1e-6,
        atol=1e-9
    )

    return sol

def solve_air_ODE(half_pipe_radius, g, mass, C_d, A, rho, state0, t_span=(0.0, 10.0)):

    # state0 = [x, y, z, x_dot, y_dot, z_dot]

    tol = 1e-6

    def rejoin_event(t, Y, *args):
        # Landing when the point intersects the cylindrical interior surface
        # Use projections onto local frame and signed distance to surface branch
        r = np.array([Y[0], Y[1], Y[2]])
        rx = float(np.dot(r, e_x))
        rn = float(np.dot(r, e_n))
        R = half_pipe_radius
        # If |rx|>R, there is no surface directly beneath; keep function positive
        if abs(rx) > R:
            return 1.0
        # Surface branch inside the pipe: rn_surface(rx) = R - sqrt(R^2 - rx^2)
        rn_surface = R - math.sqrt(max(R*R - rx*rx, 0.0))
        # Signed distance: positive outside (above), negative inside
        G = rn - rn_surface
        # Avoid triggering at t=0 (takeoff point)
        if t < 1e-6:
            return max(G, 1.0)
        return G - tol

    rejoin_event.terminal = True
    rejoin_event.direction = -1

    def along_limit_event(t, Y, *args):
        # Stop if along-pipe distance exceeds half the pipe length
        r = np.array([Y[0], Y[1], Y[2]])
        s_along = float(np.dot(r, e_s))
        return abs(s_along) - (half_pipe_length / 2.0)

    along_limit_event.terminal = True
    along_limit_event.direction = 0

    sol = solve_ivp(
        air_ODE_system,
        t_span,
        state0,
        args=(g, mass, C_d, A, rho),
        events=[rejoin_event, along_limit_event],
        rtol=1e-6,
        atol=1e-9,
    )

    return sol

'''Functions of the coordinates conversions'''

def contact_to_world(theta, s, theta_dot, s_dot):
    # Bottom-referenced geometry: origin at bottom centerline
    # Position on interior surface
    r = (half_pipe_radius * np.sin(theta)) * e_x \
        + s * e_s \
        + (half_pipe_radius - half_pipe_radius * np.cos(theta)) * e_n

    x = np.dot(r, world_X)
    y = np.dot(r, world_Y)
    z = np.dot(r, world_Z)

    # Velocity on interior surface
    v = (half_pipe_radius * theta_dot * np.cos(theta)) * e_x \
        + s_dot * e_s \
        + (half_pipe_radius * theta_dot * np.sin(theta)) * e_n

    x_dot = np.dot(v, world_X)
    y_dot = np.dot(v, world_Y)
    z_dot = np.dot(v, world_Z)
    
    return [x, y, z, x_dot, y_dot, z_dot]

def world_to_contact(x, y, z, x_dot, y_dot, z_dot):
    # world vectors
    r = np.array([x, y, z])
    v = np.array([x_dot, y_dot, z_dot])

    # position projections in local frame
    rx = np.dot(r, e_x)
    rn = np.dot(r, e_n)
    rs = np.dot(r, e_s)

    # theta measured from bottom using center shift (rn - R)
    theta = math.atan2(rx, rn - half_pipe_radius)
    s = rs

    # velocity projections
    vx = np.dot(v, e_x)
    vn = np.dot(v, e_n)
    vs = np.dot(v, e_s)

    # Recover theta_dot and s_dot
    theta_dot = (vx * math.cos(theta) + vn * math.sin(theta)) / half_pipe_radius
    s_dot = vs

    return [theta, s, theta_dot, s_dot]

'''Function for simulation'''




