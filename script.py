import numpy as np
import matplotlib.pyplot as plt

from functions import (
    half_pipe_radius,
    half_pipe_length,
    half_pipe_slope,
    g,
    mu,
    C_d,
    rho,
    mass,
    A,
    s_0,
    theta_0,
    s_dot_0,
    theta_dot_0,
    contact_to_world,
    world_to_contact,
    solve_contact_ODE,
    solve_air_ODE,
    e_x,
    e_n,
)


def simulate(max_airs: int = 6, max_segments: int = 20, tol: float = 1e-6):
    airtimes = []
    traj_x, traj_y, traj_z = [], [], []
    segments = []  # list of (phase, xs, ys, zs)

    # Initial contact state
    theta = theta_0
    s = s_0
    theta_dot = theta_dot_0
    s_dot = s_dot_0

    # Ensure we have motion: supply small along-pipe speed if both are zero
    if abs(s_dot) < 1e-8 and abs(theta_dot) < 1e-8:
        s_dot = 5.0

    segments_done = 0

    while segments_done < max_segments and len(airtimes) < max_airs:
        # Guard: out of bounds along pipe
        if abs(s) > 0.5 * half_pipe_length:
            break

        # CONTACT PHASE → up to lip
        sol_c = solve_contact_ODE(
            half_pipe_radius,
            half_pipe_length,
            half_pipe_slope,
            g,
            mu,
            mass,
            s,
            theta,
            s_dot,
            theta_dot,
        )

        # Append contact trajectory in world coordinates
        xs_c, ys_c, zs_c = [], [], []
        for i in range(sol_c.y.shape[1]):
            th_i, s_i, thd_i, sd_i = sol_c.y[:, i]
            x_i, y_i, z_i, *_ = contact_to_world(th_i, s_i, thd_i, sd_i)
            traj_x.append(x_i)
            traj_y.append(y_i)
            traj_z.append(z_i)
            xs_c.append(x_i)
            ys_c.append(y_i)
            zs_c.append(z_i)
        segments.append(("contact", np.array(xs_c), np.array(ys_c), np.array(zs_c)))

        # Determine which event occurred first: lip or S-limit
        t_lip = sol_c.t_events[0][0] if len(sol_c.t_events[0]) else None
        t_slim = sol_c.t_events[1][0] if len(sol_c.t_events) > 1 and len(sol_c.t_events[1]) else None

        if t_slim is not None and (t_lip is None or t_slim <= t_lip):
            # Reached along-pipe limit; stop simulation
            break

        if t_lip is None:
            # No lip reached; stop
            break

        t_takeoff = t_lip
        theta, s, theta_dot, s_dot = sol_c.y_events[0][0]

        # Convert to world for air ICs
        x0, y0, z0, xd0, yd0, zd0 = contact_to_world(theta, s, theta_dot, s_dot)

        # AIR PHASE → until landing
        sol_a = solve_air_ODE(
            half_pipe_radius,
            g,
            mass,
            C_d,
            A,
            rho,
            [x0, y0, z0, xd0, yd0, zd0],
            t_span=(0.0, 10.0),
        )

        # Append air trajectory
        xs_a = sol_a.y[0, :]
        ys_a = sol_a.y[1, :]
        zs_a = sol_a.y[2, :]
        for i in range(sol_a.y.shape[1]):
            traj_x.append(xs_a[i])
            traj_y.append(ys_a[i])
            traj_z.append(zs_a[i])
        segments.append(("air", xs_a, ys_a, zs_a))

        # Determine which terminal event occurred first
        t_land = sol_a.t_events[0][0] if len(sol_a.t_events[0]) else None
        t_limit = sol_a.t_events[1][0] if len(sol_a.t_events) > 1 and len(sol_a.t_events[1]) else None

        # If along-pipe limit was hit first or exclusively, stop simulation
        if t_limit is not None and (t_land is None or t_limit <= t_land):
            break

        if t_land is None:
            # Never landed back in pipe
            break

        airtimes.append(t_land - 0.0)  # air solver starts at 0.0

        # Landing state → project velocity onto surface tangent, then convert to contact
        xL, yL, zL, xdL, ydL, zdL = sol_a.y_events[0][0]
        r_vec = np.array([xL, yL, zL])
        rx = float(np.dot(r_vec, e_x))
        rn = float(np.dot(r_vec, e_n))
        theta_land = float(np.arctan2(rx, rn - half_pipe_radius))

        # Surface outward normal at contact: n_hat = sinθ e_x + cosθ e_n
        n_hat = np.sin(theta_land) * e_x + np.cos(theta_land) * e_n
        n_hat = n_hat / np.linalg.norm(n_hat)

        vL = np.array([xdL, ydL, zdL])
        v_tangent = vL - np.dot(vL, n_hat) * n_hat

        theta, s, theta_dot, s_dot = world_to_contact(xL, yL, zL, v_tangent[0], v_tangent[1], v_tangent[2])

        # Out-of-bounds guard across pipe: ensure |rx| ≤ R
        r_vec = np.array([xL, yL, zL])
        rx = np.dot(r_vec, e_x)
        if abs(rx) > half_pipe_radius + tol:
            break

        segments_done += 1

    return airtimes, np.array(traj_x), np.array(traj_y), np.array(traj_z), segments


if __name__ == "__main__":
    airtimes, xs, ys, zs, segments = simulate()

    print("Air times (s):")
    for i, at in enumerate(airtimes):
        print(f"  Jump {i+1}: {at:.4f}")

    # Plot 3D trajectory
    fig = plt.figure(figsize=(12, 4))
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    # Plot by phase: contact default color, air in red
    for phase, sx, sy, sz in segments:
        color = "red" if phase == "air" else "blue"
        ax3d.plot(sx, sy, sz, lw=0.6, color=color)
    ax3d.set_xlabel("X (across pipe)")
    ax3d.set_ylabel("Y (along pipe)")
    ax3d.set_zlabel("Z (vertical)")
    ax3d.set_title("3D trajectory")

    # 2D projections
    ax_xy = fig.add_subplot(1, 3, 2)
    for phase, sx, sy, _ in segments:
        color = "red" if phase == "air" else "blue"
        ax_xy.plot(sx, sy, lw=0.6, color=color)
    ax_xy.set_xlabel("X")
    ax_xy.set_ylabel("Y")
    ax_xy.set_title("XY projection")

    ax_xz = fig.add_subplot(1, 3, 3)
    for phase, sx, _, sz in segments:
        color = "red" if phase == "air" else "blue"
        ax_xz.plot(sx, sz, lw=0.6, color=color)
    ax_xz.set_xlabel("X")
    ax_xz.set_ylabel("Z")
    ax_xz.set_title("XZ projection")

    plt.tight_layout()
    plt.show()







