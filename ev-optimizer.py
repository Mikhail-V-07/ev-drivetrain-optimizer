"""
EV Drivetrain Optimizer v2 — SOC1020 Project 3
===============================================
Updates over v1:
  - Rolling resistance fully integrated into force balance and P_car
  - Effective eta now accounts for rolling resistance separately from gear losses
  - Default ramp angle updated to 1.6 degrees (confirmed track)
  - Default wheel radius updated to 60 mm
  - Default gear ratio updated to 7 (single/two-stage spur)
  - Rolling resistance coefficient table added for different wheel types
  - Force budget breakdown: shows ramp force vs rolling resistance separately
  - Back-calculation of real eta from a single experimental measurement
  - All summaries show rolling resistance as fraction of total force

Motor specs (from datasheet):
  No-load speed    : 3500 rpm
  Stall torque     : 2.51 mN·m
  Stall current    : 0.39 A
  Voltage          : 6 V
  Max power point  : T = 1.255 mN·m, omega = 183.3 rad/s, P = 0.23 W

Physics model (with rolling resistance):
  F_ramp   = m*g*sin(theta)                          [ramp climbing force]
  F_roll   = mu_r * m*g*cos(theta)                   [rolling resistance force]
  F_total  = F_ramp + F_roll                         [total load at wheels]
  T_motor  = F_total * r / (G * eta_gear)            [torque seen by motor]
  omega_m  = omega0 * (1 - T_motor / T_stall)        [motor speed]
  v        = (omega_m / G) * r                       [vehicle speed]
  P_car    = F_ramp * v                              [useful output power — lifting only]
  P_waste  = F_roll * v                              [power wasted to rolling resistance]

Key result (ideal, no rolling resistance):
  P_car_max = eta_gear * omega0 * T_stall / 4        [cancels G and r]

Key result (with rolling resistance):
  G and r no longer cancel — rolling resistance breaks the symmetry.
  Larger r reduces F_roll, increasing effective power fraction going to useful work.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider

# ── Motor constants ────────────────────────────────────────────────────────────
RPM_TO_RADS = 2 * np.pi / 60
G_GRAVITY   = 9.81

MOTOR = {
    "omega0" : 3500 * RPM_TO_RADS,   # no-load angular velocity [rad/s]
    "T_stall": 2.51e-3,              # stall torque [N·m]
    "I_stall": 0.39,                 # stall current [A]
    "V"      : 6.0,                  # supply voltage [V]
}

# ── Rolling resistance coefficients (for reference) ───────────────────────────
# These are approximate values. Measure your own with a simple coast-down test.
MU_R_TABLE = {
    "hard PLA on smooth asphalt"   : 0.007,
    "hard PLA on rough asphalt"    : 0.012,
    "thin rubber coat on PLA"      : 0.015,
    "soft rubber on asphalt"       : 0.025,
    "pneumatic tire on asphalt"    : 0.013,
}

# ── Core physics functions ─────────────────────────────────────────────────────

def motor_speed(T_motor):
    """Angular velocity of motor shaft given torque load [rad/s]."""
    return MOTOR["omega0"] * (1.0 - T_motor / MOTOR["T_stall"])


def motor_power(T_motor):
    """Mechanical output power of motor shaft [W]."""
    return T_motor * motor_speed(T_motor)


def forces(m, theta_rad, mu_r):
    """
    Returns (F_ramp, F_roll, F_total) in Newtons.
    F_ramp  : force needed to climb ramp
    F_roll  : rolling resistance force (all wheels combined)
    F_total : total force motor drivetrain must provide at the wheel
    """
    F_ramp  = m * G_GRAVITY * np.sin(theta_rad)
    F_roll  = mu_r * m * G_GRAVITY * np.cos(theta_rad)
    F_total = F_ramp + F_roll
    return F_ramp, F_roll, F_total


def motor_torque_from_load(m, theta_rad, r, G, eta_gear, mu_r):
    """
    Torque motor must produce to overcome ramp + rolling resistance.
    eta_gear : gear mesh efficiency only (not including rolling resistance)
    mu_r     : rolling resistance coefficient
    """
    _, _, F_total = forces(m, theta_rad, mu_r)
    return F_total * r / (G * eta_gear)


def vehicle_speed(m, theta_rad, r, G, eta_gear, mu_r):
    """Vehicle linear speed on the ramp [m/s]. Returns 0 if motor stalls."""
    T_m = motor_torque_from_load(m, theta_rad, r, G, eta_gear, mu_r)
    if np.isscalar(T_m):
        if T_m >= MOTOR["T_stall"]:
            return 0.0
    omega_m   = motor_speed(T_m)
    omega_w   = omega_m / G
    return np.where(T_m >= MOTOR["T_stall"], 0.0, omega_w * r) \
           if not np.isscalar(T_m) else omega_w * r


def vehicle_power_useful(m, theta_rad, r, G, eta_gear, mu_r):
    """
    Useful output power: only the ramp-climbing component.
    P_car = F_ramp * v   (this is what gets measured in the experiment)
    """
    F_ramp, _, _ = forces(m, theta_rad, mu_r)
    v = vehicle_speed(m, theta_rad, r, G, eta_gear, mu_r)
    return F_ramp * v


def vehicle_power_total(m, theta_rad, r, G, eta_gear, mu_r):
    """
    Total mechanical power at wheels (ramp + rolling resistance).
    P_total = F_total * v
    """
    _, _, F_total = forces(m, theta_rad, mu_r)
    v = vehicle_speed(m, theta_rad, r, G, eta_gear, mu_r)
    return F_total * v


def optimal_mass(theta_rad, r, G, eta_gear, mu_r):
    """
    Mass that maximises P_car (useful power).
    With rolling resistance, the optimum shifts lower than the ideal formula.
    Derived numerically since the rolling resistance breaks the clean analytic form.
    Falls back to analytic solution when mu_r = 0.
    """
    if mu_r == 0:
        # clean analytic result
        return (G * eta_gear * MOTOR["T_stall"]) / \
               (2 * r * G_GRAVITY * np.sin(theta_rad))
    # numerical optimum
    m_arr = np.linspace(0.005, 3.0, 5000)
    Pc    = vehicle_power_useful(m_arr, theta_rad, r, G, eta_gear, mu_r)
    return m_arr[np.argmax(Pc)]


def optimal_gear_ratio(m, theta_rad, r, eta_gear, mu_r):
    """
    Gear ratio that puts motor at half-stall torque for given mass.
    With rolling resistance, uses total force (not just ramp force).
    """
    _, _, F_total = forces(m, theta_rad, mu_r)
    return (2 * F_total * r) / (eta_gear * MOTOR["T_stall"])


def rolling_resistance_penalty(theta_rad, mu_r):
    """
    Fraction of total force that is wasted to rolling resistance.
    At theta=1.6 deg this is significant — shows why wheel choice matters.
    """
    F_ramp_unit = G_GRAVITY * np.sin(theta_rad)   # per kg
    F_roll_unit = mu_r * G_GRAVITY * np.cos(theta_rad)  # per kg
    return F_roll_unit / (F_ramp_unit + F_roll_unit)


def backsolve_eta(m_kg, t_seconds, theta_rad, r, G, mu_r,
                  track_length=10.0, track_height=None):
    """
    Given one experimental result (mass, time), back-calculate real eta_gear.
    Uses the measured P_car = mgh/t and inverts the model.

    m_kg         : measured vehicle mass [kg]
    t_seconds    : measured time to finish [s]
    theta_rad    : ramp angle [rad]
    r            : wheel radius [m]
    G            : gear ratio used
    mu_r         : rolling resistance coefficient (estimate)
    track_length : ramp length [m] (default 10 m from spec)
    track_height : ramp height [m] (computed from length and angle if None)
    """
    if track_height is None:
        track_height = track_length * np.sin(theta_rad)

    # What we measured
    P_car_measured = m_kg * G_GRAVITY * track_height / t_seconds
    v_measured     = track_length / t_seconds

    # From v = (omega0 * r / G) * (1 - F_total*r / (G*eta*T_stall))
    # Solve for eta:
    #   v * G / (omega0 * r) = 1 - F_total*r / (G * eta * T_stall)
    #   F_total*r / (G * eta * T_stall) = 1 - v*G/(omega0*r)
    #   eta = F_total*r / (G * T_stall * (1 - v*G/(omega0*r)))

    _, _, F_total = forces(m_kg, theta_rad, mu_r)
    lhs = 1.0 - (v_measured * G) / (MOTOR["omega0"] * r)

    if lhs <= 0:
        print("  ⚠ Measured speed exceeds no-load speed — check inputs.")
        return None

    eta_real = (F_total * r) / (G * MOTOR["T_stall"] * lhs)

    print(f"\n── Back-solve for real η ──")
    print(f"  Measured P_car      : {P_car_measured:.4f} W")
    print(f"  Measured speed      : {v_measured:.3f} m/s")
    print(f"  Back-solved η_gear  : {eta_real:.3f}  ({eta_real*100:.1f}%)")
    if eta_real > 0.95:
        print("  ⚠ η > 0.95 — likely mu_r is underestimated, or measurement error")
    elif eta_real < 0.30:
        print("  ⚠ η < 0.30 — very lossy gearbox, check lubrication and fit")
    return eta_real


# ── Summary printout ───────────────────────────────────────────────────────────

def print_summary(G, r_mm, theta_deg, eta_gear_pct, mu_r, m_kg=None):
    r        = r_mm / 1000
    theta    = np.radians(theta_deg)
    eta_gear = eta_gear_pct / 100

    m_opt   = optimal_mass(theta, r, G, eta_gear, mu_r)
    G_opt   = optimal_gear_ratio(m_opt, theta, r, eta_gear, mu_r)
    P_car   = vehicle_power_useful(m_opt, theta, r, G, eta_gear, mu_r)
    v       = vehicle_speed(m_opt, theta, r, G, eta_gear, mu_r)
    T_m     = motor_torque_from_load(m_opt, theta, r, G, eta_gear, mu_r)
    penalty = rolling_resistance_penalty(theta, mu_r)
    F_r, F_roll, F_tot = forces(m_opt, theta, mu_r)

    # Ideal (no rolling resistance, no gear loss) ceiling
    P_motor_max = motor_power(MOTOR["T_stall"] / 2)

    print("\n" + "═" * 58)
    print("  DRIVETRAIN SUMMARY  (with rolling resistance)")
    print("═" * 58)
    print(f"  Gear ratio G           : {G:.1f} : 1")
    print(f"  Wheel radius r         : {r_mm:.0f} mm")
    print(f"  Ramp angle θ           : {theta_deg}°")
    print(f"  Gear efficiency η_gear : {eta_gear_pct:.0f}%")
    print(f"  Rolling resistance μ_r : {mu_r:.4f}")
    print(f"  G / r                  : {G/r:.0f} m⁻¹")
    print("─" * 58)
    print(f"  Rolling loss fraction  : {penalty*100:.1f}% of total force")
    print(f"  (F_ramp={F_r:.4f} N,  F_roll={F_roll:.4f} N  @ m_opt)")
    print("─" * 58)
    print(f"  Optimal mass m_opt     : {m_opt*1000:.1f} g")
    print(f"  Optimal G_opt          : {G_opt:.1f} : 1")
    print(f"  Motor torque @ m_opt   : {T_m*1000:.3f} mN·m  "
          f"({'OK ✓' if T_m < MOTOR['T_stall'] else 'STALL ✗'})")
    print(f"  Vehicle speed          : {v:.3f} m/s")
    print(f"  Useful P_car           : {P_car:.4f} W")
    print(f"  Motor P_max (ceiling)  : {P_motor_max:.4f} W")
    print(f"  Capture fraction       : {P_car/P_motor_max*100:.1f}% of motor ceiling")

    if m_kg is not None:
        Pc_a  = vehicle_power_useful(m_kg, theta, r, G, eta_gear, mu_r)
        T_a   = motor_torque_from_load(m_kg, theta, r, G, eta_gear, mu_r)
        v_a   = vehicle_speed(m_kg, theta, r, G, eta_gear, mu_r)
        stall = T_a >= MOTOR["T_stall"]
        print("─" * 58)
        print(f"  CUSTOM MASS = {m_kg*1000:.0f} g")
        print(f"  Motor torque    : {T_a*1000:.3f} mN·m  "
              f"{'⚠ STALL' if stall else ''}")
        print(f"  Speed           : {v_a:.3f} m/s")
        print(f"  Useful P_car    : {Pc_a:.4f} W")
    print("═" * 58 + "\n")


def print_mu_r_table(theta_deg):
    """Show rolling resistance penalty for each wheel type at given ramp angle."""
    theta = np.radians(theta_deg)
    print(f"\n── Rolling resistance penalty at θ = {theta_deg}° ──")
    print(f"  {'Wheel type':<35} {'μ_r':>6}  {'% of total force':>16}")
    print("  " + "─" * 60)
    for name, mu in MU_R_TABLE.items():
        pen = rolling_resistance_penalty(theta, mu) * 100
        print(f"  {name:<35} {mu:>6.4f}  {pen:>14.1f}%")
    print()


# ── Static four-panel analysis ─────────────────────────────────────────────────

def plot_static_analysis(G=7.0, r=0.060, theta_deg=1.6, eta_gear=0.82, mu_r=0.010):
    theta = np.radians(theta_deg)

    # Motor curve
    T_range  = np.linspace(0, MOTOR["T_stall"], 300)
    P_motor  = motor_power(T_range)
    T_opt_m  = MOTOR["T_stall"] / 2
    P_max_m  = motor_power(T_opt_m)

    # P_car vs mass
    m_opt   = optimal_mass(theta, r, G, eta_gear, mu_r)
    m_range = np.linspace(0.005, max(m_opt * 2.5, 1.0), 400)
    P_car_m = vehicle_power_useful(m_range, theta, r, G, eta_gear, mu_r)
    Pc_opt  = vehicle_power_useful(m_opt, theta, r, G, eta_gear, mu_r)

    # P_car vs gear ratio (at m_opt)
    G_range = np.linspace(1, 30, 300)
    P_car_g = vehicle_power_useful(m_opt, theta, r, G_range, eta_gear, mu_r)
    G_opt   = optimal_gear_ratio(m_opt, theta, r, eta_gear, mu_r)

    # Sensitivity: P_car(m) for different mu_r values
    mu_r_vals = [0.005, 0.010, 0.015, 0.025]
    mu_colors = ["#3aaa6e", "#4a90d9", "#e07b39", "#e05252"]

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        f"EV Analysis  |  G={G:.0f}  r={r*1000:.0f}mm  "
        f"θ={theta_deg}°  η_gear={eta_gear*100:.0f}%  μ_r={mu_r:.3f}",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)

    # Panel 1: motor curve + operating point
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(T_range * 1e3, P_motor, color="#4a90d9", lw=2, label="Motor P(T)")
    ax1.axvline(T_opt_m * 1e3, ls="--", color="#4a90d9", lw=1, alpha=0.6,
                label=f"Motor P_max = {P_max_m:.3f} W")
    T_op = motor_torque_from_load(m_opt, theta, r, G, eta_gear, mu_r)
    if T_op < MOTOR["T_stall"]:
        ax1.scatter([T_op * 1e3], [motor_power(T_op)], color="#e05252",
                    zorder=5, s=80, label=f"Op. point @ m_opt={m_opt*1000:.0f}g")
    ax1.set_xlabel("Motor torque (mN·m)")
    ax1.set_ylabel("Power (W)")
    ax1.set_title("Motor output power curve")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # Panel 2: P_car vs mass
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(m_range * 1000, P_car_m, color="#e07b39", lw=2, label="Useful P_car(m)")
    ax2.axvline(m_opt * 1000, ls="--", color="#e07b39", lw=1, alpha=0.7,
                label=f"m_opt = {m_opt*1000:.1f} g")
    ax2.scatter([m_opt * 1000], [Pc_opt], color="#e05252", zorder=5, s=80,
                label=f"P_car = {Pc_opt:.4f} W")
    # Show rolling resistance fraction
    pen = rolling_resistance_penalty(theta, mu_r)
    ax2.set_title(f"P_car vs mass  (roll loss = {pen*100:.1f}% of total force)")
    ax2.set_xlabel("Vehicle mass (g)")
    ax2.set_ylabel("Useful output power (W)")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # Panel 3: P_car vs gear ratio
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(G_range, P_car_g, color="#7b5ea7", lw=2)
    ax3.axvline(G_opt, ls="--", color="#7b5ea7", lw=1, alpha=0.7,
                label=f"G_opt = {G_opt:.1f}")
    ax3.axvline(G, ls=":", color="#aaa", lw=1.2, label=f"Current G = {G:.0f}")
    Pc_Gopt = vehicle_power_useful(m_opt, theta, r, G_opt, eta_gear, mu_r)
    if 1 < G_opt < 30:
        ax3.scatter([G_opt], [Pc_Gopt], color="#e05252", zorder=5, s=80,
                    label=f"P = {Pc_Gopt:.4f} W")
    ax3.set_xlabel("Gear ratio G")
    ax3.set_ylabel("Useful P_car (W)")
    ax3.set_title(f"P_car vs G  (m = m_opt = {m_opt*1000:.1f} g)")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    # Panel 4: sensitivity to rolling resistance
    ax4 = fig.add_subplot(gs[1, 1])
    for mu, col in zip(mu_r_vals, mu_colors):
        Pc_s  = vehicle_power_useful(m_range, theta, r, G, eta_gear, mu)
        m_o   = optimal_mass(theta, r, G, eta_gear, mu)
        Pc_o  = vehicle_power_useful(m_o, theta, r, G, eta_gear, mu)
        pen_s = rolling_resistance_penalty(theta, mu) * 100
        ax4.plot(m_range * 1000, Pc_s, color=col, lw=1.8,
                 label=f"μ_r={mu:.3f}  (roll={pen_s:.0f}%)")
        if 0 < m_o < m_range[-1]:
            ax4.scatter([m_o * 1000], [Pc_o], color=col, zorder=5, s=40)
    ax4.set_xlabel("Vehicle mass (g)")
    ax4.set_ylabel("Useful P_car (W)")
    ax4.set_title("Sensitivity to rolling resistance coefficient")
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

    plt.savefig("ev_analysis_v2.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: ev_analysis_v2.png")


# ── Interactive slider plot ────────────────────────────────────────────────────

def plot_interactive():
    fig, ax = plt.subplots(figsize=(9, 6))
    plt.subplots_adjust(left=0.10, bottom=0.50, right=0.95, top=0.93)

    m_arr = np.linspace(0.005, 2.0, 600)

    def compute(G, r_mm, theta_deg, eta_pct, mu_r):
        r     = r_mm / 1000
        theta = np.radians(theta_deg)
        eta   = eta_pct / 100
        Pc    = vehicle_power_useful(m_arr, theta, r, G, eta, mu_r)
        m_o   = optimal_mass(theta, r, G, eta, mu_r)
        Pc_o  = float(vehicle_power_useful(m_o, theta, r, G, eta, mu_r))
        G_o   = optimal_gear_ratio(m_o, theta, r, eta, mu_r)
        pen   = rolling_resistance_penalty(theta, mu_r) * 100
        T_m   = motor_torque_from_load(m_o, theta, r, G, eta, mu_r)
        return Pc, m_o, Pc_o, G_o, pen, T_m

    G0, r0, th0, et0, mu0 = 7.0, 60.0, 1.6, 82.0, 0.010
    Pc0, m_o0, Pc_o0, G_o0, pen0, T0 = compute(G0, r0, th0, et0, mu0)

    [line_Pc] = ax.plot(m_arr * 1000, Pc0, color="#e07b39", lw=2.2,
                        label="Useful P_car(m)")
    [pt_opt]  = ax.plot([m_o0 * 1000], [Pc_o0], "o", color="#e05252", ms=9,
                        label=f"m_opt = {m_o0*1000:.1f} g")
    P_ceil = motor_power(MOTOR["T_stall"] / 2)
    ax.axhline(P_ceil, color="#4a90d9", ls="--", lw=1.2, alpha=0.6,
               label=f"Motor ceiling = {P_ceil:.3f} W")

    ax.set_xlabel("Vehicle mass (g)", fontsize=11)
    ax.set_ylabel("Useful output power (W)", fontsize=11)
    ax.set_title("Interactive drivetrain optimizer v2", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, P_ceil * 1.15)

    info = ax.text(0.98, 0.97, "", transform=ax.transAxes, fontsize=9,
                   va="top", ha="right",
                   bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc", alpha=0.9))

    # Sliders
    sl_specs = [
        ("sl_G",   [0.10, 0.38, 0.78, 0.025], "Gear ratio G",     1,  20,  G0,  0.5),
        ("sl_r",   [0.10, 0.32, 0.78, 0.025], "Wheel r (mm)",    20,  70,  r0,  1.0),
        ("sl_th",  [0.10, 0.26, 0.78, 0.025], "Ramp θ (°)",      0.5, 10,  th0, 0.1),
        ("sl_eta", [0.10, 0.20, 0.78, 0.025], "η_gear (%)",      30,  95,  et0, 1.0),
        ("sl_mu",  [0.10, 0.14, 0.78, 0.025], "μ_r ×1000",        2,  30,  mu0*1000, 0.5),
    ]
    sliders = {}
    for name, rect, label, vmin, vmax, vinit, vstep in sl_specs:
        sax = plt.axes(rect)
        sliders[name] = Slider(sax, label, vmin, vmax, valinit=vinit, valstep=vstep)

    def update(_):
        G   = sliders["sl_G"].val
        r   = sliders["sl_r"].val
        th  = sliders["sl_th"].val
        eta = sliders["sl_eta"].val
        mu  = sliders["sl_mu"].val / 1000.0

        Pc, m_o, Pc_o, G_o, pen, T_m = compute(G, r, th, eta, mu)
        stall = T_m >= MOTOR["T_stall"]

        line_Pc.set_ydata(Pc)
        if 0 < m_o < 2.0 and Pc_o > 0 and not stall:
            pt_opt.set_data([m_o * 1000], [Pc_o])
        else:
            pt_opt.set_data([], [])

        info.set_text(
            f"m_opt   = {m_o*1000:.1f} g\n"
            f"G_opt   = {G_o:.1f} : 1\n"
            f"P_car   = {Pc_o:.4f} W\n"
            f"G / r   = {G / (sliders['sl_r'].val/1000):.0f} m⁻¹\n"
            f"Roll %  = {pen:.1f}% of force\n"
            f"Motor T = {T_m*1e3:.3f} mN·m\n"
            f"{'⚠ STALL' if stall else 'OK ✓'}"
        )
        ax.legend(fontsize=9)
        fig.canvas.draw_idle()

    for sl in sliders.values():
        sl.on_changed(update)
    update(None)

    plt.suptitle("EV Optimizer v2 — includes rolling resistance", fontsize=11, y=0.99)
    plt.show()


# ── Experimental data fitting ──────────────────────────────────────────────────

def fit_experimental_data(masses_g, powers_W, degree=2, label="experiment"):
    """
    Fit polynomial to measured (mass, P_car) data.
    Also back-calculates eta for each data point if speed data provided.
    """
    m = np.array(masses_g) / 1000
    P = np.array(powers_W)

    coeffs = np.polyfit(m, P, degree)
    p_fit  = np.poly1d(coeffs)

    m_dense   = np.linspace(m.min() * 0.5, m.max() * 1.5, 1000)
    P_dense   = p_fit(m_dense)
    m_opt_exp = m_dense[np.argmax(P_dense)]
    P_opt_exp = P_dense.max()

    if degree == 2 and coeffs[0] < 0:
        m_opt_analytic = -coeffs[1] / (2 * coeffs[0])
    else:
        m_opt_analytic = None

    print(f"\n── Experimental fit: {label} ──")
    print(f"  Coefficients (deg {degree}): {coeffs}")
    print(f"  Empirical m_opt : {m_opt_exp*1000:.1f} g")
    if m_opt_analytic is not None:
        print(f"  Analytic m_opt  : {m_opt_analytic*1000:.1f} g  (-b/2a)")
    print(f"  Max P_car       : {P_opt_exp:.4f} W")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(m * 1000, P, color="#e05252", zorder=5, s=70, label="Measured")
    ax.plot(m_dense * 1000, p_fit(m_dense), color="#4a90d9", lw=2,
            label=f"Poly deg={degree}")
    ax.axvline(m_opt_exp * 1000, ls="--", color="#4a90d9", lw=1, alpha=0.7,
               label=f"m_opt = {m_opt_exp*1000:.1f} g")
    ax.scatter([m_opt_exp * 1000], [P_opt_exp], color="#e07b39", zorder=6, s=90,
               label=f"P_max = {P_opt_exp:.4f} W")
    ax.set_xlabel("Vehicle mass (g)")
    ax.set_ylabel("P_car (W)")
    ax.set_title(f"Experimental fit — {label}")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"fit_{label.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")
    return coeffs, m_opt_exp, P_opt_exp


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Design parameters (edit these) ──
    G        = 7        # gear ratio
    R_MM     = 60       # wheel radius in mm
    THETA    = 1.6      # ramp angle in degrees
    ETA_GEAR = 82       # gear efficiency in percent (two spur stages ~0.95^2 = 0.90,
                        # but include bearing losses -> ~0.82 is a reasonable estimate)
    MU_R     = 0.010    # rolling resistance: hard PLA on asphalt

    # 1. Show rolling resistance penalty table
    print_mu_r_table(THETA)

    # 2. Full summary with rolling resistance
    print_summary(G, R_MM, THETA, ETA_GEAR, MU_R)

    # 3. Static four-panel analysis
    plot_static_analysis(G=G, r=R_MM/1000, theta_deg=THETA,
                         eta_gear=ETA_GEAR/100, mu_r=MU_R)

    # 4. Interactive slider plot
    plot_interactive()

    # 5. Back-solve eta from ONE real experiment
    #    Replace with your actual measurement after your first test run
    #    Example: car mass 380g, finished 10m track in 8.5 seconds
    # backsolve_eta(
    #     m_kg=0.380, t_seconds=8.5,
    #     theta_rad=np.radians(1.6), r=0.060, G=7, mu_r=0.010
    # )

    # 6. Fit experimental data (replace with real readings)
    example_masses = [150, 250, 350, 450, 550, 650]   # grams
    example_powers = [0.04, 0.09, 0.13, 0.14, 0.13, 0.10]  # Watts
    fit_experimental_data(example_masses, example_powers, degree=2,
                          label="G7 r60mm theta1.6")
