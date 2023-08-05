import numpy as np

"""
Configuration file
Permits various adjustments to parameters of the FastSLAM algorithm.
See fastslam_sim.py for more information
"""

# control parameters
V = 3 # m/s
MAXG = 30*np.pi/180 # radians, maximum steering angle (-MAXG < g < MAXG)
RATEG = 20*np.pi/180 # rad/s, maximum rate of change in steer angle
WHEELBASE = 4 # metres, vehicle wheel-base
DT_CONTROLS = 0.025 # seconds, time interval between control signals

# control noises
sigmaV = 0.3 # m/s
sigmaG = (3.0*np.pi/180) # radians
Q = np.array([[sigmaV**2, 0], [0, sigmaG**2]])

# observation parameters
MAX_RANGE = 30.0 # metres
DT_OBSERVE = 8*DT_CONTROLS # seconds, time interval between observations

# observation noises
sigmaR = 0.1 # metres
sigmaB = (1.0*np.pi/180) # radians
R = np.array([[sigmaR**2, 0], [0, sigmaB**2]])

# waypoint proximity
AT_WAYPOINT = 1.0 # metres, distance from current waypoint at which to switch to next waypoint
NUMBER_LOOPS = 1 # number of loops through the waypoint list

# resampling
NPARTICLES= 100; 
NEFFECTIVE= 0.75*NPARTICLES; # minimum number of effective particles before resampling

# switches
SWITCH_CONTROL_NOISE = 1 # if 0, velocity and gamma are perfect
SWITCH_SENSOR_NOISE = 1 # if 0, measurements are perfect
SWITCH_INFLATE_NOISE = 0 # if 1, the estimated Q and R are inflated (ie, add stabilising noise)
SWITCH_PREDICT_NOISE = 1; # sample noise from predict (usually 1 for fastslam1.0 and 0 for fastslam2.0)
SWITCH_HEADING_KNOWN = 0 # if 1, the vehicle heading is observed directly at each iteration
SWITCH_RESAMPLE= 1; 
SWITCH_SEED_RANDOM= 0 # if not 0, seed the randn() with its value at beginning of simulation (for repeatability)
