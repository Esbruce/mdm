import numpy as np
import math
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import logging

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

s_0 = 0
theta_0 = 0.0
s_dot_0 = 1
theta_dot_0 = 2

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
e_n = np.array([0, -np.sin(half_pipe_slope), np.cos(half_pipe_slope)])

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

    # Drag acceleration components (safe even if speed=0)
    x_ddot = -k * speed * x_dot
    y_ddot = -g - k * speed * y_dot
    z_ddot = -k * speed * z_dot

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

    # Events detection

    # Stop when theta reaches +pi/2 (stop_at="pos") or -pi/2 (stop_at!="pos")
    target_theta = math.pi / 2 if str(stop_at).lower() == "pos" else -math.pi / 2

    def theta_limit_event(t, y, *args):
        # y = [theta, s, theta_dot, s_dot]
        return y[0] - target_theta

    theta_limit_event.terminal = True
    theta_limit_event.direction = 0  # detect crossing in any direction

    # Solve the ODE

    sol = solve_ivp(
        contact_ODE_system,
        t_span,
        y_0,
        args=(half_pipe_radius, half_pipe_length, half_pipe_slope, g, mu, mass),
        events=theta_limit_event,
        rtol=1e-6,
        atol=1e-9
    )

    return sol

def solve_air_ODE(half_pipe_radius, g, mass, C_d, A, rho, state0, t_span=(0.0, 10.0)):

    # state0 = [x, y, z, x_dot, y_dot, z_dot]

    def y_surface(x):
        R = half_pipe_radius
        if abs(x) > R:
            return -np.inf
        return R - math.sqrt(max(R*R - x*x, 0.0))

    def rejoin_event(t, Y, *args):
        x, y = Y[0], Y[1]
        return y - y_surface(x)

    rejoin_event.terminal = True
    rejoin_event.direction = -1

    sol = solve_ivp(
        air_ODE_system,
        t_span,
        state0,
        args=(g, mass, C_d, A, rho),
        events=rejoin_event,
        rtol=1e-6,
        atol=1e-9,
    )

    return sol

'''Functions of the coordinates conversions'''

def contact_to_world(theta, s, theta_dot, s_dot):
    
    r = (half_pipe_radius * np.sin(theta)) * e_x + s * e_s + (half_pipe_radius * np.cos(theta)) * np.array([0, np.sin(half_pipe_slope), np.cos(half_pipe_slope)])

    x = np.dot(r, world_X)      
    y = np.dot(r, world_Y)
    z = np.dot(r, world_Z)

    v = (half_pipe_radius * theta_dot * np.cos(theta)) * e_x + s_dot * e_s - (half_pipe_radius * theta_dot * np.sin(theta)) * np.array([0, np.sin(half_pipe_slope), np.cos(half_pipe_slope)])

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

    theta = np.arctan2(rx, rn)
    s = rs

    # velocity projections

    vx = np.dot(v, e_x)  
    vn = np.dot(v, e_n)  
    vs = np.dot(v, e_s)  

    theta_dot = (vx * np.cos(theta) - vn * np.sin(theta)) / half_pipe_radius
    s_dot = vs

    return [theta, s, theta_dot, s_dot]

'''Function for simulation'''




def run_simulation(max_air_phases=50,air_t_span=(0.0, 60.0),log_level=logging.INFO):
    """
    Runs alternating contact and air phases starting from global initial conditions.

    Returns:
        total_airtime (float): Sum of durations of all air phases [s]
        num_air_phases (int): Number of completed air phases
    """

    logger = logging.getLogger("simulation")
    if not logger.handlers:
        # Leave handler configuration to caller; ensure logger exists
        pass
    logger.setLevel(log_level)

    def is_out_of_bounds(x_value):
        # Consider out-of-bounds if |x| exceeds pipe radius by tolerance
        return abs(x_value) > (half_pipe_radius + 1e-6)

    airtimes = []

    # Start from contact initial conditions (global parameters)
    current_theta = theta_0
    current_s = s_0
    current_theta_dot = theta_dot_0
    current_s_dot = s_dot_0

    # Determine which side to target first based on current theta sign
    current_side = "pos" if current_theta >= 0.0 else "neg"

    for phase_index in range(max_air_phases):
        logger.info("Contact phase %d: side=%s", phase_index + 1, current_side)

        # CONTACT → run until takeoff angle boundary reached
        contact_sol = solve_contact_ODE(
            half_pipe_radius,
            half_pipe_length,
            half_pipe_slope,
            g,
            mu,
            mass,
            s_0=current_s,
            theta_0=current_theta,
            s_dot_0=current_s_dot,
            theta_dot_0=current_theta_dot,
            stop_at=current_side,
        )

        if len(contact_sol.t) == 0 or len(contact_sol.y.T) == 0:
            logger.warning("No contact integration data; terminating simulation.")
            break

        # Takeoff state at the terminal event (last state)
        theta_tk, s_tk, theta_dot_tk, s_dot_tk = contact_sol.y[:, -1]
        logger.info(
            "Takeoff at t=%.3f s: theta=%.3f rad, s=%.3f m, theta_dot=%.3f, s_dot=%.3f",
            contact_sol.t[-1], theta_tk, s_tk, theta_dot_tk, s_dot_tk,
        )

        # Convert to world coordinates to start AIR phase
        x0, y0, z0, xdot0, ydot0, zdot0 = contact_to_world(
            theta_tk, s_tk, theta_dot_tk, s_dot_tk
        )

        if is_out_of_bounds(x0):
            logger.info("Takeoff out of bounds (x=%.3f); terminating.", x0)
            break

        # AIR → integrate until rejoin with surface
        air_state0 = [x0, y0, z0, xdot0, ydot0, zdot0]
        air_sol = solve_air_ODE(
            half_pipe_radius,
            g,
            mass,
            C_d,
            A,
            rho,
            air_state0,
            t_span=air_t_span,
        )

        if air_sol.t_events and len(air_sol.t_events[0]) > 0:
            airtime = air_sol.t_events[0][0]
        else:
            # No rejoin detected in window; treat as out of bounds
            airtime = air_sol.t[-1] if len(air_sol.t) > 0 else 0.0
            logger.warning("No landing detected within air window; terminating after t=%.3f s.", airtime)
            airtimes.append(max(airtime, 0.0))
            break

        airtimes.append(max(airtime, 0.0))
        logger.info("Air phase %d duration: %.3f s", phase_index + 1, airtime)

        # Landing state is last integrated state
        xL, yL, zL, xdotL, ydotL, zdotL = air_sol.y[:, -1]

        if is_out_of_bounds(xL):
            logger.info("Landing out of bounds (x=%.3f); terminating.", xL)
            break

        # Convert landing world state back to contact variables for next loop
        theta_L, s_L, theta_dot_L, s_dot_L = world_to_contact(xL, yL, zL, xdotL, ydotL, zdotL)
        current_theta, current_s, current_theta_dot, current_s_dot = theta_L, s_L, theta_dot_L, s_dot_L

        # Alternate side for next contact phase
        current_side = "neg" if current_side == "pos" else "pos"

    total_airtime = float(sum(airtimes))
    num_air_phases = int(len(airtimes))

    logger.info("Simulation complete: phases=%d, total airtime=%.3f s", num_air_phases, total_airtime)

    return total_airtime, num_air_phases

