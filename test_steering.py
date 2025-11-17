import numpy as np
import math
from parameters import half_pipe_radius

# Test the steering functions with direction awareness
def s_from_theta_test(theta, c1, c3, c4, direction_sign):
    x = half_pipe_radius * math.sin(theta)
    s_desired = c1 * (math.tanh(c3 * x + c4) + math.tanh(direction_sign * c3 * 10.0))
    return s_desired

def s_dot_from_theta_test(theta, theta_dot, c1, c3, c4, direction_sign):
    x = half_pipe_radius * math.sin(theta)
    ds_dtheta = c1 * c3 * half_pipe_radius * math.cos(theta) * (1 / math.cosh(c3 * x + c4))**2
    s_dot_desired = ds_dtheta * theta_dot
    return s_dot_desired

# Test parameters from C[0]
c1, c2, c3, c4 = 20, 1, 0.1, 0

print("Testing steering functions:")
print(f"Parameters: c1={c1}, c3={c3}, c4={c4}")
print()

# Test at right lip (theta = pi/2)
theta = np.pi/2
theta_dot = -1.0  # moving left (decreasing theta)

print(f"At right lip (theta={theta:.3f}, theta_dot={theta_dot}):")
s_pos = s_from_theta_test(theta, c1, c3, c4, +1.0)
s_neg = s_from_theta_test(theta, c1, c3, c4, -1.0)
print(f"  s_desired (direction_sign=+1): {s_pos:.3f}")
print(f"  s_desired (direction_sign=-1): {s_neg:.3f}")
print()

# Test at center (theta = 0)
theta = 0.0
theta_dot = -1.0

print(f"At center (theta={theta:.3f}, theta_dot={theta_dot}):")
s_pos = s_from_theta_test(theta, c1, c3, c4, +1.0)
s_neg = s_from_theta_test(theta, c1, c3, c4, -1.0)
sdot_pos = s_dot_from_theta_test(theta, theta_dot, c1, c3, c4, +1.0)
sdot_neg = s_dot_from_theta_test(theta, theta_dot, c1, c3, c4, -1.0)
print(f"  s_desired (direction_sign=+1): {s_pos:.3f}")
print(f"  s_desired (direction_sign=-1): {s_neg:.3f}")
print(f"  s_dot_desired (direction_sign=+1): {sdot_pos:.3f}")
print(f"  s_dot_desired (direction_sign=-1): {sdot_neg:.3f}")
print()

# The derivative should be the same regardless of direction_sign
# because the offset term is constant
print("Note: s_dot_desired is THE SAME for both directions")
print("This is CORRECT because tanh(direction_sign*c3*10) is constant")

