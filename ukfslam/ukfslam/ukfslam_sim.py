import numpy as np
from . import configfile as c # ** USE THIS FILE TO CONFIGURE THE UKF-SLAM **
from .compute_steering import compute_steering
from .vehicle_model import vehicle_model
from .add_control_noise import add_control_noise
from .predict import predict
from .observe_heading import observe_heading
from .get_observations import get_observations
from .add_observation_noise import add_observation_noise
from .data_associate_known import data_associate_known
from .update import update
from .augment import augment

def ukfslam_sim(lm, wp, phi):
    # Initialise states and other global variables
    xtrue = np.zeros((3,1))
    xtrue[2] = phi
    XX = np.zeros((3,1))
    XX[2] = phi
    PX = np.eye(3)*np.finfo(float).eps
    data = initialise_store(XX,PX,xtrue) # stored data for off-line

    # Initialise other variables and constants
    dt = c.DT_CONTROLS # change in time between predicts
    dtsum = 0 # change in time since last observation
    ftag = np.arange(lm.shape[1]) # identifier for each landmark
    da_table = np.zeros((1, lm.shape[1])) - 1 # data association table 
    iwp = 0 # index to first waypoint 
    G = 0 # initial steer angle
    QE = np.copy(c.Q)
    RE = np.copy(c.R)
    if c.SWITCH_INFLATE_NOISE:
        QE = np.copy(2*c.Q)
        RE = np.copy(2*c.R)
    if c.SWITCH_SEED_RANDOM:
        np.random.seed(c.SWITCH_SEED_RANDOM)

    NUMBER_LOOPS = c.NUMBER_LOOPS

    # Main loop 
    while iwp != -1:

        # Compute true data
        G,iwp = compute_steering(xtrue, wp, iwp, c.AT_WAYPOINT, G, c.RATEG, c.MAXG, dt)
        if iwp==-1 and NUMBER_LOOPS > 1:
            iwp=0
            NUMBER_LOOPS= NUMBER_LOOPS-1 # perform loops: if final waypoint reached, go back to first
        xtrue = vehicle_model(xtrue, c.V,G, c.WHEELBASE,dt)
        Vn,Gn = add_control_noise(c.V,G,c.Q, c.SWITCH_CONTROL_NOISE)

        # UKF predict step
        XX, PX = predict (XX, PX, Vn,Gn,QE, c.WHEELBASE,dt)

        # If heading known, observe heading
        XX, PX = observe_heading(XX, PX, xtrue[2][0], c.SWITCH_HEADING_KNOWN)

        # Incorporate observation, (available every DT_OBSERVE seconds)
        dtsum = dtsum + dt
        if dtsum >= c.DT_OBSERVE:
            dtsum = 0

            # Compute true data
            z,ftag_visible = get_observations(xtrue, lm, ftag, c.MAX_RANGE)
            z = add_observation_noise(z,c.R, c.SWITCH_SENSOR_NOISE)

            # UKF update step
            zf,idf,zn, da_table = data_associate_known(XX,z,ftag_visible, da_table)

            XX, PX = update(XX, PX, zf,RE,idf) 
            XX, PX = augment(XX, PX, zn,RE) 

        # Offline data store
        data = store_data(data, XX, PX, xtrue)

    # end of main loop
    data = finalise_data(data)
    return data


def initialise_store(x, P, xtrue):
    # offline storage initialisation
    data = {}
    data['i'] = 1
    data['path'] = np.zeros((3, 1))
    data['path'][0,0] = x[0][0]
    data['path'][1,0] = x[1][0]
    data['path'][2,0] = x[2][0]
    data['true'] = np.zeros((3, 1))
    data['true'][0,0] = xtrue[0][0]
    data['true'][1,0] = xtrue[1][0]
    data['true'][2,0] = xtrue[2][0]
    data['state'] = [{}]
    data['state'][0]['x'] = np.copy(x)
    data['state'][0]['PV'] = np.copy(P)
    data['state'][0]['P'] = np.diag(np.copy(P))
    return data 

def store_data(data, x, P, xtrue):    
    # add current data to offline storage
    CHUNK = 5000
    if data['i'] == data['path'].shape[1]: # grow array in chunks to amortise reallocation
        data['path'] = np.hstack((data['path'], np.zeros((3, CHUNK))))
        data['true'] = np.hstack((data['true'], np.zeros((3, CHUNK))))
    i = data['i'] + 1
    data['i'] = i
    data['path'][0,i-1] = x[0][0]
    data['path'][1,i-1] = x[1][0]
    data['path'][2,i-1] = x[2][0]
    data['true'][0,i-1] = xtrue[0][0]
    data['true'][1,i-1] = xtrue[1][0]
    data['true'][2,i-1] = xtrue[2][0]
    data['state'].append({})
    data['state'][i-1]['x'] = np.copy(x)
    #data['state'][i-1]['P'] = P
    data['state'][i-1]['P'] = np.diag(P)
    data['state'][i-1]['PV'] = np.copy(P[0:3,0:3])
    return data

def finalise_data(data):
    # offline storage finalisation
    data['path'] = data['path'][:, 0:data['i']]
    data['true'] = data['true'][:, 0:data['i']]
    return data
