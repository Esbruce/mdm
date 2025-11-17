import numpy as np
import math
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from math import pi


''' Simulation Parameters '''

from parameters import (
    half_pipe_radius,
    half_pipe_length,
    half_pipe_slope,
    g,
    mu,
    C_d,
    rho,
    mass,
    A,
    k_p_default,
    k_d_default,
    u_s_max_default,
    s_0,
    theta_0,
    s_dot_0,
    theta_dot_0,
    x_0,
    y_0,
    z_0,
    x_dot_0,
    y_dot_0,
    z_dot_0,

    e_x,
    e_s,
    e_n,
    world_X,
    world_Y,
    world_Z,

    epsilon,
    landing_tol,
)

''' Functions returning the ODEs for the contact and air phases steering included in contact'''

def contact_ODE_system(t, y, half_pipe_radius, half_pipe_length, half_pipe_slope, g, mu, mass, k_p=k_p_default, k_d=k_d_default, u_s_max=u_s_max_default, C=None, s0=0.0, s0_for_ctrl=None):
    
    # Unpack state
    theta, s, theta_dot, s_dot = y
    
    # Compute speed (avoid division by zero)

    speed = math.sqrt(s_dot**2 + half_pipe_radius**2 * theta_dot**2)
    if speed < epsilon:
        speed = epsilon

    # Natural ODE's for passive motion 

    s_ddot = g* np.sin(half_pipe_slope) - mu * (half_pipe_radius * theta_dot**2 + g * np.cos(half_pipe_slope) * np.cos(theta)) * (s_dot / speed) - ((0.5*rho* C_d*A)/ mass)*(speed)* s_dot

    theta_ddot = -((g * np.cos(half_pipe_slope) * np.sin(theta)) / half_pipe_radius) - mu * (half_pipe_radius * theta_dot**2 + g * np.cos(half_pipe_slope) * np.cos(theta)) * (theta_dot / speed) - ((0.5*rho* C_d*A)/ mass)*(speed)* theta_dot

    # In case of no steering set controller to zero

    U_s = 0.0

    if C is not None and len(C) == 4 and (abs(k_p) > 0.0 or abs(k_d) > 0.0):

        s_0 = s0_for_ctrl

        # Desired s from current theta (tanh route)

        s_desired = s_from_theta(theta, C) # returns the s we are aiming for

        s_dot_desired = s_dot_from_theta(theta, theta_dot, C) # returns the s_dot we are aiming for

        # Relative S Position this phase

        s_rel = s - s0
        
        s_desired_rel = s_desired # We get S from the y=0 line
        
        # PD control

        U_s = -k_p * (s_rel - s_desired_rel) - k_d * (s_dot - s_dot_desired)
        
        # set maximus steering force

        if u_s_max > 0.0:

            U_s = np.clip(U_s, -u_s_max, u_s_max)

    # Apply control to s only; theta remains natural

    s_ddot = s_ddot + U_s

    return [theta_dot, s_dot, theta_ddot, s_ddot]

def air_ODE_system(t, Y, g, mass, C_d, A, rho):
     
     # Unpack state vector
     x, y, z, x_dot, y_dot, z_dot = Y
 
     # Velocity magnitude
     speed = math.sqrt(x_dot * x_dot + y_dot * y_dot + z_dot * z_dot)
 
     # Drag coefficient factor
     k = 0.5 * rho * C_d * A / mass
 
     # Drag acceleration components (Z is vertical)
     x_ddot = -k * speed * x_dot
     y_ddot = -k * speed * y_dot
     z_ddot = -g - k * speed * z_dot
 
     # Return first-order derivatives: [x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot]
     return [x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot]

''' Functions for solving the ODEs with event detection incorporated.'''

def solve_contact_ODE(half_pipe_radius, half_pipe_length, half_pipe_slope, g, mu, mass, s_0, theta_0, s_dot_0, theta_dot_0, stop_at="pos", C=None, k_p=k_p_default, k_d=k_d_default, u_s_max=u_s_max_default, enable_apex=False, theta_apex_min=0.35, dt_phase_min=0.15, t_phase_max=None, s_phase_max=None, s_phase_start=None):

    # Initial conditions
    theta = theta_0
    s = s_0
    theta_dot = theta_dot_0
    s_dot = s_dot_0

    y_0 = [theta, s, theta_dot, s_dot] # packaging the intial conditions 

    # Time span (cap by per-phase max if provided)
    t_max = float(t_phase_max) if t_phase_max is not None else 1000.0
    t_span = (0, t_max) # time we integrate over (max)

    '''Event detections'''

    # Takeoff when reaching either lip |theta| = pi/2 

    def lip_event(t, y, *args):

        theta = y[0] # extracts theta from the package

        return abs(theta) - (math.pi / 2) # if this equals 0 then the rider is at the lip

    lip_event.terminal = True
    lip_event.direction = 1  # crossing from inside toward the lip
   

    # Along-pipe limit: stop if |s| exceeds half the pipe length

    def s_limit_event(t, y, *args):
        s = y[1] # extracts the s from the package
        # Trigger when reaching either end of the pipe: |s| = half_pipe_length
        return abs(s) - (half_pipe_length)

    s_limit_event.terminal = True
    s_limit_event.direction = 0

    # Optional: apex-based rollover event (theta_dot crosses zero) with guards
    def apex_event(t, y, *args):

        theta = y[0]
        theta_dot = y[2]

        # guard against chattering at bottom and too-early triggers
        if abs(theta) < float(theta_apex_min) or t < float(dt_phase_min):
            # keep positive to avoid zero crossing detection
            return 1.0

        return theta_dot

    if enable_apex:
        apex_event.terminal = True
        apex_event.direction = 0

    # Optional: per-phase along-distance cap
    def s_phase_event(t, y, *args):
        if s_phase_max is None or s_phase_start is None:
            return 1.0
        s = y[1]
        return abs(s - float(s_phase_start)) - float(s_phase_max)
    if s_phase_max is not None and s_phase_start is not None:
        s_phase_event.terminal = True
        s_phase_event.direction = 0


    # Optional: per-phase time cap
    def t_limit_event(t, y, *args):
        if t_phase_max is None:
            return 1.0
        return t - float(t_phase_max)

    if t_phase_max is not None:
        t_limit_event.terminal = True
        t_limit_event.direction = 1

    # Stop if speed drops too low (rider essentially stopped)
    def stopped_event(t, y, *args):
        theta_dot = y[2]
        s_dot = y[3]
        speed = math.sqrt(s_dot**2 + (half_pipe_radius * theta_dot)**2)
        return speed - 0.5  # m/s threshold
    
    stopped_event.terminal = True
    stopped_event.direction = -1  # trigger when crossing downward

    '''Solve the ODE'''

    events_list = [lip_event, s_limit_event, stopped_event]
    if enable_apex:
        events_list.append(apex_event)
    if s_phase_max is not None and s_phase_start is not None:
        events_list.append(s_phase_event)
    if t_phase_max is not None:
        events_list.append(t_limit_event)


    # steering local origin for this phase

    s0_for_ctrl = s_phase_start if s_phase_start is not None else s
    theta0_for_ctrl = theta 

    sol = solve_ivp(
        contact_ODE_system,
        t_span,
        y_0,
        args=(half_pipe_radius, half_pipe_length, half_pipe_slope, g, mu, mass, k_p, k_d, u_s_max, C, s0_for_ctrl, theta0_for_ctrl),
        events=events_list,
        rtol=1e-6,
        atol=1e-9,
        max_step=0.1  # Prevent infinitely small steps when system becomes stiff
    )

    return sol

def solve_air_ODE(half_pipe_radius,g,mass,C_d,A, rho, state0, t_span=(0.0, 10.0), t_eval=None, max_step=None, rtol=1e-6, atol=1e-9, dense_output=True):

    # state0 = [x, y, z, x_dot, y_dot, z_dot]

    '''Event detection'''

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
        return G - landing_tol

    rejoin_event.terminal = True
    rejoin_event.direction = -1

    def along_limit_event(t, Y, *args):
        # Stop if along-pipe distance exceeds the pipe length
        r = np.array([Y[0], Y[1], Y[2]])
        s_along = float(np.dot(r, e_s))
        return s_along - half_pipe_length

    along_limit_event.terminal = True
    along_limit_event.direction = 0

    # Ensure valid numeric step/tolerance parameters for SciPy
    # Set reasonable default max_step for reliable event detection (10ms)

    max_step_val = 0.01 if max_step is None else float(max_step)
    rtol_val = float(rtol)
    atol_val = float(atol)

    sol = solve_ivp(
        air_ODE_system,
        t_span,
        state0,
        args=(g, mass, C_d, A, rho),
        events=[rejoin_event, along_limit_event],
        rtol=rtol_val,
        atol=atol_val,
        max_step=max_step_val,
        t_eval=t_eval,
        dense_output=dense_output,
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

'''Steering helpers'''

def _clamp(value, lo, hi):
    return max(lo, min(hi, value))

def s_from_theta(theta, C_phase):
    # check there is correct coefficients

    if len(C_phase) < 4:
        raise ValueError("C_tanh must contain at least [c1, c2, c3, c4] for the tanh route (c2 is ignored).")

    c1, _, c3, c4 = [float(C_phase[i]) for i in range(4)]  # extract coefficients for phase passed in

    # Guard against degenerate parameters
    if abs(c3) < 1e-12:
        # Flat mapping: S = 0 (centered)
        return 0.0

    # Across-pipe coordinate from theta
    
    x = half_pipe_radius * math.sin(theta)



    # Route function: S(x) = c1 * (tanh(c3 * x) + tanh(c3 * 10 * sign(c3)))

    s_desired = c1 * (math.tanh(c3 * x) + math.tanh(abs(c3) * 10.0)) # correct

    return s_desired

def s_dot_from_theta(theta, theta_dot, C_phase):

    if len(C_phase) < 4:
        raise ValueError("C_tanh must contain at least [c1, c2, c3, c4] for the tanh route (c2 is ignored).")
        
    c1, _ , c3, _ = [float(C_phase[i]) for i in range(4)]  # extract coefficients for phase passed in
    
    # Guard against degenerate parameters
    if abs(c3) < 1e-12:
        # Flat mapping: S = 0 (centered)
        return 0.0
    
    ds_dtheta = c1 * c3 * half_pipe_radius * np.cos(theta)* (1 / np.cosh(c3 * half_pipe_radius * np.sin(theta)))**2

    s_dot_desired = ds_dtheta * theta_dot
    
    return s_dot_desired




def generate_initial_velocities_and_intial_S(speed, theta_0, C_phase, direction='decreasing'):

    if speed <= 0:
        raise ValueError(f"Speed must be positive, got {speed}")
    
    if half_pipe_radius <= 0:
        raise ValueError(f"Half-pipe radius must be positive, got {half_pipe_radius}")
    
    # Determine sign based on direction
    sign = 1.0 if direction == 'increasing' else -1.0


    c1, _, c3, c4 = [float(C_phase[i]) for i in range(4)]

    ds_dtheta = c1 * c3 * half_pipe_radius * np.cos(theta_0) * (1 / np.cosh(c3 * half_pipe_radius * np.sin(theta_0)))**2

    # Resolve components from total speed
    denom = math.sqrt(half_pipe_radius**2 + ds_dtheta**2)

    if denom == 0.0:
        raise ZeroDivisionError("degenerate geometry: R and ds/dθ both zero")

    theta_dot = sign * (speed / denom)
    s_dot = ds_dtheta * theta_dot

    s0 = s_from_theta(theta_0, C_phase)


    return theta_dot, s_dot, s0






