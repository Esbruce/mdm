import numpy as np
import matplotlib.pyplot as plt

from parameters import (
    half_pipe_radius,
    half_pipe_length,
    half_pipe_slope,
    theta_0,
    g,
    mu,
    C_d,
    rho,
    mass,
    A,
    k_p,
    k_d,
    u_s_max,
    s_0,
    theta_0,
    e_x,
    e_s,
    e_n,
    world_X,
    world_Y,
    world_Z,
    epsilon,
    max_segments,
    max_airs,
    use_steering,
    s_phase_max,
    t_phase_max,
    t_air_phase_max,
    air_r_tol,
    air_a_tol,
    landing_tol
)

from logic import (
    contact_to_world,
    generate_initial_velocities_and_intial_S,
    world_to_contact,
    solve_contact_ODE,
    solve_air_ODE,
    generate_initial_velocities_and_intial_S,
    s_from_theta,
    s_dot_from_theta,
)



C = [
    [20, 1, -0.1, 0],
    [20, 1,  0.1, 0],
    [20, 1, -0.1, 0],
    [20, 1,  0.1, 0],
    [20, 1, -0.1, 0],
]

# Generate initial velocities and position from desired speed and theta_0

initial_speed = 15.0  # m/s
if initial_speed > 0.0:
    theta_dot_0, s_dot_0, s0 = generate_initial_velocities_and_intial_S(
        speed=initial_speed, theta_0=theta_0, C_phase=C[0], direction="decreasing"
    )
else:
    theta_dot_0, s_dot_0 = 0.0, 0.0
    # Align initial s with the route when speed is zero
    s0 = s_from_theta(theta_0, C[0])


# Initialise lists for tracking

airtimes = []

traj_x = []
traj_y = []
traj_z = []

segments = []

total_turning_effort = 0.0
phase_id = 0
t_contact_accum = 0.0
route_id = -1
segments_done = 0
start_new_contact = True
start_new_air = False

# Initialise Variables for last time

s=s0
theta=theta_0
theta_dot = theta_dot_0
s_dot = s_dot_0


while segments_done < max_segments and len(airtimes) < max_airs:

    # check s position is within pipe 

    if abs(s) > half_pipe_length:
        break 


    # Track progress to avoid infinite loop if no new points are added
    _prev_points = len(traj_x)

    if start_new_contact == True:

        

        route_id += 1

        phase_id +=1

        print(f'Phase = {phase_id}')

        print('Solving contact ODE')

        sol_contact = solve_contact_ODE(
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
        C=C[route_id] if use_steering else None,
        k_p=k_p if use_steering else 0.0,
        k_d=k_d if use_steering else 0.0,
        u_s_max=u_s_max,
        s_phase_start=s,
        s_phase_max=s_phase_max,
        t_phase_max=t_phase_max
        )

        

        # Store contact trajectory

        xs_c, ys_c, zs_c = [], [], []
        for i in range(sol_contact.y.shape[1]):
            th_i, s_i, thd_i, sd_i = sol_contact.y[:, i]
            x_i, y_i, z_i, *_ = contact_to_world(th_i, s_i, thd_i, sd_i)
            traj_x.append(x_i)
            traj_y.append(y_i)
            traj_z.append(z_i)
            xs_c.append(x_i)
            ys_c.append(y_i)
            zs_c.append(z_i)
        segments.append(("contact", np.array(xs_c), np.array(ys_c), np.array(zs_c)))

        # print(f'Contact Phase trajectory: x:{xs_c} y:{ys_c} Z:{zs_c}')

        # Determine earliest event
        event_names = ["lip", "s_limit", "stopped"]
        if s_phase_max is not None:
            event_names.append("phase_S")
        if t_phase_max is not None:
            event_names.append("phase_t")

        first_event_idx = None
        first_event_time = None
        for idx, times in enumerate(sol_contact.t_events):
            if len(times) and (first_event_time is None or times[0] < first_event_time):
                first_event_time = times[0]
                first_event_idx = idx

        if first_event_idx is None:
            print(f"[no_event] No event detected in contact phase; exiting.")
            break

        first_event_name = event_names[first_event_idx] if first_event_idx < len(event_names) else "unknown"
        th_ev, s_ev, thd_ev, sd_ev = sol_contact.y_events[first_event_idx][0]
        
        print(f"[event] {first_event_name} at theta={th_ev:.3f}, s={s_ev:.3f}")
        
        # Check if rider has essentially stopped (very low speed)
        speed_at_event = np.sqrt(sd_ev**2 + (half_pipe_radius * thd_ev)**2)
        if speed_at_event < 0.5:  # m/s threshold
            print(f"[stopped] Speed too low ({speed_at_event:.3f} m/s); exiting.")
            break

        if first_event_name == "s_limit":
            print(f"[s_limit]")
            break
        
        if first_event_name == "stopped":
            print(f"[stopped] Rider came to a stop; exiting.")
            break

        if first_event_name == "lip":
            # Prepare to enter air phase
            theta, s, theta_dot, s_dot = th_ev, s_ev, thd_ev, sd_ev
            start_new_contact = False
            start_new_air = True
            phase_id += 1
        elif first_event_name in ["phase_S", "phase_t"]:
            # Intra-contact phase transition (time/distance cap)
            theta, s, theta_dot, s_dot = th_ev, s_ev, thd_ev, sd_ev
            phase_id += 1
            # If we've exhausted all phases in the C list, stop
            if phase_id >= len(C):
                print(f"[phase_exhausted] Reached end of steering phases; exiting.")
                break
            
            # Check progress before continuing (safety guard)
            if len(traj_x) == _prev_points:
                print("[stop] No new trajectory points added; exiting loop.")
                break
            
            start_new_contact = True
            continue
        else:
            # Unknown event - shouldn't happen but break to be safe
            print(f"[unknown_event] {first_event_name}; exiting.")
            break


    if start_new_air == True:

        # Convert to world coordinates for air phase

        x, y, z, x_dot, y_dot, z_dot = contact_to_world(theta, s, theta_dot, s_dot)
        
        # Solve air ODE

        t_span = (0, t_air_phase_max)

        t_eval = None

        print(f'Phase={phase_id}')

        sol_air = solve_air_ODE(
            half_pipe_radius,
            g,
            mass,
            C_d,
            A,
            rho,
            [x, y, z, x_dot, y_dot, z_dot],
            t_span=t_span,
            rtol= air_r_tol,
            atol= air_a_tol,
        )

        # Store air trajectory
        xs_a = sol_air.y[0, :]
        ys_a = sol_air.y[1, :]
        zs_a = sol_air.y[2, :]
        for i in range(sol_air.y.shape[1]):
            traj_x.append(xs_a[i])
            traj_y.append(ys_a[i])
            traj_z.append(zs_a[i])
        segments.append(("air", xs_a, ys_a, zs_a))


        # Check landing event
        t_land = sol_air.t_events[0][0] if len(sol_air.t_events[0]) else None
        t_limit = (
            sol_air.t_events[1][0] if len(sol_air.t_events) > 1 and len(sol_air.t_events[1]) else None
        )


        if t_limit is not None and (t_land is None or t_limit <= t_land):
            print(f"[along_limit]")
            phase_id += 1
            break

        if t_land is None:
            print(f"[no_land]")
            phase_id += 1
            break

        airtimes.append(t_land)
        x_land, y_land, z_land, xdv_land, ydv_land, zdv_land = sol_air.y_events[0][0]

        
        print(f"[landing] phase={phase_id} air_time={t_land:.4f}")

        # Project landing velocity onto surface tangent
        r_vec = np.array([x_land, y_land, z_land])
        rx = float(np.dot(r_vec, e_x))
        rn = float(np.dot(r_vec, e_n))
        theta_land = float(np.arctan2(rx, rn - half_pipe_radius))
        n_hat = np.sin(theta_land) * e_x + np.cos(theta_land) * e_n
        n_hat = n_hat / np.linalg.norm(n_hat)
        vL = np.array([xdv_land, ydv_land, zdv_land])
        v_tangent = vL - np.dot(vL, n_hat) * n_hat

        # Convert back to contact coordinates
        theta, s, theta_dot, s_dot = world_to_contact(
            x_land, y_land, z_land, v_tangent[0], v_tangent[1], v_tangent[2]
        )

        # Check bounds
        if abs(np.dot(np.array([x_land, y_land, z_land]), e_x)) > half_pipe_radius + landing_tol:
            break

        segments_done += 1
        start_new_air = False
        start_new_contact = True



    # Break if no new trajectory points were added this iteration (safety guard)
    if len(traj_x) == _prev_points:
        print("[stop] No new trajectory points added; exiting loop.")
        break

    # plotting moved outside the simulation loop

print(F'total airtime: {sum(airtimes):.4f} seconds')

# Single 3D trajectory plot (entire run)
if len(traj_x) >= 1:
    all_x = np.array(traj_x)
    all_y = np.array(traj_y)
    all_z = np.array(traj_z)

    # Determine s-range for rendering surface
    rs_traj = np.stack([all_x, all_y, all_z], axis=1)
    s_proj = rs_traj @ e_s
    s_min_traj = float(np.min(s_proj))
    s_max_traj = float(np.max(s_proj))
    s_margin = 0.05 * (s_max_traj - s_min_traj) if s_max_traj > s_min_traj else 1.0
    s_min = max(-half_pipe_length, s_min_traj - s_margin)
    s_max = min(half_pipe_length, s_max_traj + s_margin)

    fig3d = plt.figure(figsize=(10, 10))
    ax3d = fig3d.add_subplot(111, projection="3d")

    # Half-pipe surface
    R = float(half_pipe_radius)
    n_theta, n_s = 60, 120
    Theta, Sgrid = np.meshgrid(
        np.linspace(-np.pi / 2, np.pi / 2, n_theta),
        np.linspace(s_min, s_max, n_s),
        indexing="ij",
    )
    sinT, cosT = np.sin(Theta), np.cos(Theta)
    r_grid = (
        (R * sinT)[..., None] * e_x
        + Sgrid[..., None] * e_s
        + (R - R * cosT)[..., None] * e_n
    )
    Xsurf, Ysurf, Zsurf = r_grid[..., 0], r_grid[..., 1], r_grid[..., 2]
    ax3d.plot_surface(
        Xsurf, Ysurf, Zsurf, color="lightgray", alpha=0.25, linewidth=0, antialiased=True
    )

    # Plot trajectory segments with different colors for air vs contact
    if len(segments) > 0:
        for seg_type, xs, ys, zs in segments:
            color = "red" if seg_type == "air" else "blue"
            if len(xs) >= 2:
                ax3d.plot(xs, ys, zs, lw=0.8, color=color)
            elif len(xs) == 1:
                ax3d.scatter(xs, ys, zs, s=20, color=color)
    else:
        # Fallback to original behavior if segments list is empty
        if all_x.size >= 2:
            ax3d.plot(all_x, all_y, all_z, lw=0.8, color="blue")
        else:
            ax3d.scatter(all_x, all_y, all_z, s=20, color="blue")

    # Set axis limits and aspect
    x_range = float(np.max(all_x) - np.min(all_x))
    y_range = float(np.max(all_y) - np.min(all_y))
    z_range = float(np.max(all_z) - np.min(all_z))
    x_pad = x_range * 0.05 if x_range > 0 else 1.0
    y_pad = y_range * 0.05 if y_range > 0 else 1.0
    z_pad = z_range * 0.05 if z_range > 0 else 1.0
    ax3d.set_xlim(float(np.min(all_x) - x_pad), float(np.max(all_x) + x_pad))
    ax3d.set_ylim(float(np.min(all_y) - y_pad), float(np.max(all_y) + y_pad))
    ax3d.set_zlim(float(np.min(all_z) - z_pad), float(np.max(all_z) + z_pad))
    ax3d.set_xlabel("X (across pipe)")
    ax3d.set_ylabel("Y (along pipe)")
    ax3d.set_zlabel("Z (vertical)")
    ax3d.set_title("3D trajectory")

    max_extent = max(x_range + 2 * x_pad, y_range + 2 * y_pad, z_range + 2 * z_pad)
    if max_extent > 0:
        box_aspect = [
            (x_range + 2 * x_pad) / max_extent,
            (y_range + 2 * y_pad) / max_extent,
            (z_range + 2 * z_pad) / max_extent,
        ]
        ax3d.set_box_aspect(box_aspect)
    else:
        ax3d.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()

    # 2D x–s plot (x horizontal, s vertical)
    x_traj = rs_traj @ e_x
    s_traj = s_proj  # already computed
    fig2d, ax2d = plt.subplots(figsize=(8, 8))
    # Plot 2D trajectory segments with different colors for air vs contact
    if len(segments) > 0:
        for seg_type, xs, ys, zs in segments:
            color = "red" if seg_type == "air" else "blue"
            x_seg = np.array([np.dot([x, y, z], e_x) for x, y, z in zip(xs, ys, zs)])
            s_seg = np.array([np.dot([x, y, z], e_s) for x, y, z in zip(xs, ys, zs)])
            if x_seg.size >= 2:
                ax2d.plot(x_seg, s_seg, color=color, lw=0.8)
            else:
                ax2d.plot(x_seg, s_seg, 'o', color=color, markersize=4)
    else:
        # Fallback to original behavior if segments list is empty
        if x_traj.size >= 2:
            ax2d.plot(x_traj, s_traj, color="blue", lw=0.8)
        else:
            ax2d.plot(x_traj, s_traj, 'o', color="blue", markersize=4)
    # Pipe bounds: vertical lines at x = ±R
    ax2d.axvline(x=half_pipe_radius, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax2d.axvline(x=-half_pipe_radius, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax2d.set_xlabel("x (across pipe)")
    ax2d.set_ylabel("s (along pipe)")
    ax2d.set_title("x–s trajectory")
    ax2d.set_aspect("auto")
    plt.tight_layout()
    plt.show()




