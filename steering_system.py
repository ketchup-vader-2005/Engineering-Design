import numpy as np
from scipy.optimize import minimize

# --- VEHICLE INPUTS ---
A = 2650.0      # Wheelbase (mm)
E = 1530.0      # Wheel track (mm)
PHI_V_MAX = 540 # Steering wheel rotation (degrees) to one side (1.5 turns)
X_MAX = 190.0    # Rack stroke (mm) - Defined here so it's accessible everywhere

def solve_kinematics(params, A, E, x_val):
    l, t, yc = params
    ly0 = np.arctan((E/2) / A)
    lx0 = np.radians(90) - ly0
    
    # Neutral pos calculation for tie rod length 'b'
    xA, yA = E/2, 0
    xB = xA - l * np.sin(ly0)
    yB = l * np.cos(ly0)
    b = np.sqrt(((t/2) - xB)**2 + (yc - yB)**2)
    
    angles = []
    for disp in [x_val, -x_val]: 
        di = (E/2) - (t/2) + disp
        P, Q = 2 * di * l, 2 * yc * l
        R_const = b**2 - di**2 - yc**2 - l**2
        
        a_q, b_q, c_q = (R_const - P), 2*Q, (P + R_const)
        delta = b_q**2 - 4 * a_q * c_q
        if delta < 0: return None, None, b
        
        z = (-b_q - np.sqrt(delta)) / (2 * a_q)
        v = np.degrees(lx0 - 2 * np.arctan(z))
        angles.append(v)
    return angles[0], angles[1], b

def radius_error_objective(params):
    l, t, yc = params
    
    # Derive the resulting dp. If dp < 20, we penalize.
    dp = (X_MAX * 360) / (np.pi * PHI_V_MAX)
    
    if dp < 20.0:
        return 1e6 # Pinion too small
    
    errors = []
    # Evaluate Ackermann Radius Error
    for x in np.linspace(5, X_MAX, 10):
        vi, ve, _ = solve_kinematics(params, A, E, x)
        if vi is None or vi <= ve: return 1e6
        
        vi_rad, ve_rad = np.radians(vi), np.radians(ve)
        # Avoid division by zero if angles are too small/equal
        denom = (np.tan(vi_rad) - np.tan(ve_rad))
        if abs(denom) < 1e-9: return 1e6
        
        R = (E/2) * ((np.tan(ve_rad) + np.tan(vi_rad)) / denom)
        e = abs(A - (R - E/2) * np.tan(vi_rad))
        errors.append(e)
        
    return np.mean(errors)

# --- OPTIMIZATION ---
# [l, t, yc]
initial_guess = [146.0, 764.0, 105.0] 
res = minimize(radius_error_objective, initial_guess, bounds=[(100, 200), (600, 950), (50, 250)])

if res.success:
    l_opt, t_opt, yc_opt = res.x
    dp_opt = (X_MAX * 360) / (np.pi * PHI_V_MAX)
    _, _, b_opt = solve_kinematics(res.x, A, E, 0)
    
    print(f"--- RESULTS (Pinion > 20mm) ---")
    print(f"Optimal Arm (l):    {l_opt:.2f} mm")
    print(f"Optimal Rack (t):   {t_opt:.2f} mm")
    print(f"Optimal Offset (yc):{yc_opt:.2f} mm")
    print(f"Tie Rod (b):        {b_opt:.2f} mm")
    print(f"Pinion (dp):        {dp_opt:.2f} mm") 
    print(f"Rack Stroke (x_val):{X_MAX:.2f} mm") # <--- Added x_val output
    print(f"Mean Radius Error:  {res.fun:.4f} mm")
else:
    print("Optimization failed:", res.message)