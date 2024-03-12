"""This module models the orbit of a spacecraft around planets with restricted 
motion

ALL NUMBERS in km, rad, s"""
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize

def gravity_accel(craft_pos, planet_pos, planet_mu):
    """Find the acceleration vector of spacecraft due to given planet
    
    Inputs: craft_pos  -- 1D array of position vector of spacecraft
            planet_pos -- 1D array of position vector of planet
            planet_mu  -- float graviational parameter of planet
    Outputs: craft_accel -- 1D array of acceleration vector of spacecraft
    """
    craft_planet_pos = planet_pos - craft_pos
    if np.linalg.norm(craft_planet_pos) < 500:
        raise ValueError("Spacecraft orbit is unstable / has crashed")
    craft_planet_dist = np.linalg.norm(craft_planet_pos)
    craft_accel = planet_mu * craft_planet_pos / craft_planet_dist**3
    return craft_accel

def body_pos_vel(t, body):
    """Find position and velocity of a celestial body at t sec past J2000 epoch
    
    Inputs: t    -- float seconds since J2000 epoch
            body -- string name of celestial body 
    Outputs: planet_pos -- 1D array of position vector of planet
             planet_vel -- 1D array of velocity vector of planet
    """
    if body == "sun":
        return np.zeros(2), np.zeros(2)
    mu = ephem.at["sun", "mu"]
    omega_bar = ephem.at[body, 'omega_bar']
    omega_bar_dot = ephem.at[body, 'omega_bar_dot']
    omega_bar += omega_bar_dot * t
    Omega = ephem.at[body, 'Omega']
    Omega_dot = ephem.at[body, 'Omega_dot']
    Omega += Omega_dot * t
    omega = omega_bar - Omega
    L = ephem.at[body, 'L']
    L_dot = ephem.at[body, 'L_dot']
    L += L_dot * t
    M = L - omega_bar
    a = ephem.at[body, 'a']
    a_dot = ephem.at[body, 'a_dot']
    a += a_dot * t
    e = ephem.at[body, 'e']
    e_dot = ephem.at[body, 'e_dot']
    e += e_dot * t
    h = (mu * a * (1 - e**2))**.5
    E = Kepler(M, e)
    theta = 2 * np.arctan(((1 + e) / (1 - e))**.5 * np.tan(E / 2))
    r_pf = h**2 / mu / (1 + e * np.cos(theta))
    r_pf *= np.array([np.cos(theta), np.sin(theta)])
    v_pf = mu / h * np.array([-np.sin(theta), e + np.cos(theta)])
    C = PF_to_Eq(omega, Omega)
    planet_pos = C @ r_pf
    planet_vel = C @ v_pf
    return planet_pos, planet_vel

def Kepler(M, e):
    """Find the eccentric anomaly
    
    Inputs: M -- scalar mean anomaly
            e -- scalar eccentricity
    Outputs: E -- scalar eccetricic anomaly
    """
    def f(E):
        return M - E + e * np.sin(E)
    def df(E):
        return -1 + e * np.cos(E)
    def df2(E):
        return -e * np.sin(E)
    E = optimize.root_scalar(f, x0=0, fprime=df, fprime2=df2).root
    return E

def PF_to_Eq(omega, Omega):
    """Find matrix to convert from perifocal frame to equatorial frame
    
    Inputs: omega -- scalar lowercase omega in radians
            Omega -- scalar uppercase Omega in radians
    Outputs: C -- 2D array
    """
    C = np.array([[np.cos(omega), np.sin(omega)],
                  [-np.sin(omega), np.cos(omega)]])
    temp = np.array([[np.cos(Omega), np.sin(Omega)],
                     [-np.sin(Omega), np.cos(Omega)]])
    C = C @ temp
    return C.T

def dudt(t, u):
    """Find derivative of u vector
    
    Inputs: t -- scalar time since start of problem
            u -- 1D array with spacecraft position in front, velocity in back
    Outputs: du -- 1D array with velocity in front, acceleration in back
    """
    if u.shape[0] != 4:
        raise ValueError("u.shape is "+str(u.shape))
    du = np.zeros_like(u)
    du[:2] = u[-2:]
    for body in ephem.index:
        body_mu = ephem.at[body,"mu"]
        du[2:] += gravity_accel(u[:2], body_pos_vel(t, body)[0], body_mu)
    return du

def crashEvent(t, u):
    """Return the distance of the spacecraft from a 500 km sphere around the 
    nearest planet, center
    
    Inputs: t -- scalar time since start of problem
            u -- 1D array with spacecraft position in front, velocity in back
    Outputs: r -- scalar distance from 500 km sphere around nearest planet's center
    """
    r = None
    for body in ephem.index:
        body_pos = body_pos_vel(t, body)[0]
        r_temp = np.linalg.norm(u[:2] - body_pos) - 500
        if r is None or r_temp < r:
            r = r_temp
    if r is None:
        raise ValueError("r is still None")
    return r

crashEvent.terminal = True

def exitEvent(t, u):
    """Return the distance of the spacecraft from Earth's SoI
    
    Inputs: t -- scalar time since start of problem
            u -- 1D array with spacecraft position in front, velocity in back
    Outputs: r -- scalar distance from Earth's SoI
    """
    r = earthSoI - np.linalg.norm(u[:2] - body_pos_vel(t, 'earth')[0])
    return r

exitEvent.terminal = True

def rot(theta):
    """Find the matrix to rotate a 2D vector theta radians CCW
    
    Inputs: theta -- scalar angle of CCW rotation in radians
    Outputs: rot_mat -- 2D array of rotation
    """
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    return rot_mat

ephem = pd.read_csv("Ephemeris.csv", index_col=0)
earth_mu = ephem.at["earth","mu"]
# mu, a (dot), e (dot), Omega (dot), w_bar (dot), L (dot)
# NOTE: All values must be in km, rad, and s

"---------------------------------------------------"
"Initial satellite conditions"
# angle between sun-earth vector and earth-craft vector at start in rad
angle_i = np.pi - 1.769
# distance of spacecraft from Earth in km
position_i = 924.9e3
# angle between earth velocity vector and craft velocity vector at start in rad
angle_v = 2 * np.pi - 1.791
# velocity relative to the earth at start  in km/s
start_v = 4
"---------------------------------------------------"

earthSoI = 925e3
# t_span = np.array([-69811200, 141955200])
# loading = np.linspace(t_span[0], t_span[1], 10001)
t_span = np.array([0, 500000])

earth_pos_i, earth_vel_i = body_pos_vel(t_span[0], 'earth')
craft_pos_i = earth_vel_i * (position_i / np.linalg.norm(earth_vel_i)) # km
craft_pos_i = earth_pos_i + rot(angle_i) @ craft_pos_i
craft_vel_i = start_v * earth_vel_i / np.linalg.norm(earth_vel_i)
craft_vel_i = earth_vel_i + rot(angle_v) @ craft_vel_i
print(earth_vel_i)

u0 = np.concatenate((craft_pos_i, craft_vel_i))

sol_t = np.array([t_span[0]])
sol_y = np.array([u0]).T

# for i in range(1000):
#     t_span = np.array([loading[i], loading[i+1]])
#     sol = solve_ivp(dudt, t_span, sol_y[:,-1].T, method="Radau",
#                     rtol=1e-5, atol=1e-7, events=(crashEvent, exitEvent))
#     sol_t = np.concatenate((sol_t, sol.t))
#     sol_y = np.concatenate((sol_y, sol.y), axis=1)
#     print(i)

sol = solve_ivp(dudt, t_span, sol_y[:,-1].T, method="Radau",
                rtol=1e-5, atol=1e-7, events=(crashEvent, exitEvent))
sol_t = np.concatenate((sol_t, sol.t))
sol_y = np.concatenate((sol_y, sol.y), axis=1)

pos = np.zeros([4, sol_t.shape[0], 2])
vel = np.zeros([4, sol_t.shape[0], 2])
for i, time in enumerate(sol_t):
    pos[0,i], vel[0,i] = body_pos_vel(time, "venus")
    pos[1,i], vel[1,i] = body_pos_vel(time, "earth")
    pos[2,i], vel[2,i] = body_pos_vel(time, "jupiter")
    pos[3,i], vel[3,i] = body_pos_vel(time, "saturn")


fig, ax = plt.subplots(1,1, figsize=(6, 4)) # 6in x 4in figure
plt.plot(sol_y[0],sol_y[1],label="spacecraft pos")
# plt.plot(pos[0,:,0],pos[0,:,1],label="venus")
plt.plot(pos[1,:,0],pos[1,:,1],label="earth")
# plt.plot(pos[2,:,0],pos[2,:,1],label="jupiter")
# plt.plot(pos[3,:,0],pos[3,:,1],label="saturn")
plt.plot(0, 0, marker='o', label='sun')
ax.set_ylabel('km')
ax.set_xlabel('km')
# plt.plot(sol.t,np.linalg.norm(sol.y[:2], axis=0),label="spacecraft pos")
# q = ax.quiver(pos[:,0], pos[:,1], vel[:,0], vel[:,1], color='b', units='xy', scale=1e-8)
# ax.quiverkey(q, X=.75, Y=.05, U=2., label="earth vel", labelpos='E')
plt.legend(loc=3)
plt.savefig('sun_orbit.png', dpi=200, edgecolor='none')

fig, ax = plt.subplots(1,1, figsize=(6, 4)) # 6in x 4in figure
plt.plot(sol_t,np.linalg.norm(sol_y[-2:], axis=0),label="spacecraft vel")
ax.set_ylabel("km/s")
ax.set_xlabel('time (s)')
vel_i = np.linalg.norm(sol_y[-2:,0], axis=0)
vel_f = np.linalg.norm(sol_y[-2:,-1], axis=0)
ax.set_title(f'vel_i={vel_i:.2f} km/s, vel_f={vel_f:.2f} km/s')
print(np.linalg.norm(sol_y[-2:,0]-sol_y[-2:,-1], axis=0))
print(np.linalg.norm(sol_y[-2:,-1]-vel[1,-1]))
print(sol_y[-2:,0])
print(sol_y[-2:,-1])
print(vel[1,0])
print(vel[1,-1])

fig, ax = plt.subplots(1,1, figsize=(6, 4)) # 6in x 4in figure
plt.plot(sol_y[0]-pos[1,:,0],sol_y[1]-pos[1,:,1],label="spacecraft pos")
plt.legend(loc=4)
ax.set_ylabel("km")
ax.set_xlabel('km')
ax.axis('equal')
fig.suptitle('Position of spacecraft relative to Earth')
# earth_vel_f = np.linalg.norm(vel[1,-1])
alpha = np.arccos(np.sum(sol_y[-2:,-1]*sol_y[-2:,0])/vel_f/vel_i)
ax.set_title(f'alpha={alpha:.2f} rad')

# plt.savefig('velocity.png', dpi=200, edgecolor='none')

# fig, ax = plt.subplots(1,1, figsize=(6, 4)) # 6in x 4in figure
# plt.plot(sol_t,np.linalg.norm(sol_y[:2]-pos[0,:].T, axis=0),label="spacecraft pos to venus")
# ax.set_ylabel("Distance from spacecraft to Venus [km]")
# ax.set_xlabel('Time from J2000 [s]')