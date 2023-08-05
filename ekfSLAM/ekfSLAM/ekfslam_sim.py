import numpy as np
from . import configfile as c # ** USE THIS FILE TO CONFIGURE THE EKF-SLAM **
from .compute_steering import compute_steering
from .vehicle_model import vehicle_model
from .add_control_noise import add_control_noise
from .predict import predict
from .observe_heading import observe_heading
from .get_observations import get_observations
from .add_observation_noise import add_observation_noise
from .data_associate_known import data_associate_known
from .data_associate import data_associate
from .update import update
from .augment import augment

def ekfslam_sim(lm, wp, phi):
    """
     data= ekfslam_sim(lm, wp, phi)

     INPUTS: 
       lm - set of landmarks
       wp - set of waypoints
       phi - initial rotation angle

     OUTPUTS:
       data - a dictonary containing:
           data['true']: the vehicle 'true'-path (ie, where the vehicle *actually* went)
           data['path']: the vehicle path estimate (ie, where SLAM estimates the vehicle went)
           data['state'][k]['x']: the SLAM state vector at time k
           data['state'][k]['P']: the diagonals of the SLAM covariance matrix at time k

     NOTES:
       This program is a SLAM simulator. To use, create a set of landmarks and 
       vehicle waypoints (ie, waypoints for the desired vehicle path).
           The configuration of the simulator is managed by the script file
       'configfile.py'. To alter the parameters of the vehicle, sensors, etc
       adjust this file. There are also several switches that control certain
       filter options.

     Version 1.0
    """

    # initialise states
    xtrue = np.zeros((3,1))
    xtrue[2] = phi
    x = np.zeros((3,1))
    x[2] = phi
    P = np.zeros((3,3))   

    # initialise other variables and constants
    dt = c.DT_CONTROLS
    dtsum = 0
    ftag = np.arange(lm.shape[1])
    da_table = np.zeros((1, lm.shape[1])) - 1
    iwp = 0
    G = 0
    data = initialise_store(x,P,xtrue)
    QE = np.copy(c.Q)
    RE = np.copy(c.R)
    if c.SWITCH_INFLATE_NOISE:
        QE = np.copy(2*c.Q)
        RE = np.copy(8*c.R)
    if c.SWITCH_SEED_RANDOM:
        np.random.seed(c.SWITCH_SEED_RANDOM)
        
    NUMBER_LOOPS = c.NUMBER_LOOPS
        
    # main loop
    while iwp != -1:    
        # compute true data
        G,iwp= compute_steering(xtrue, wp, iwp, c.AT_WAYPOINT, G, c.RATEG, c.MAXG, dt)
        if iwp== -1 & NUMBER_LOOPS > 1:
            iwp=0
            NUMBER_LOOPS= NUMBER_LOOPS-1 # perform loops: if final waypoint reached, go back to first
        xtrue= vehicle_model(xtrue, c.V,G, c.WHEELBASE,dt)
        Vn,Gn= add_control_noise(c.V,G,c.Q, c.SWITCH_CONTROL_NOISE)
    
        # EKF predict step
        x,P= predict (x,P, Vn,Gn,QE, c.WHEELBASE,dt)
    
        # if heading known, observe heading
        x,P= observe_heading(x,P, xtrue[2][0], c.SWITCH_HEADING_KNOWN)
    
        # EKF update step
        dtsum= dtsum + dt
        if dtsum >= c.DT_OBSERVE:
            dtsum= 0
            z,ftag_visible= get_observations(xtrue, lm, ftag, c.MAX_RANGE)
            z= add_observation_noise(z,c.R, c.SWITCH_SENSOR_NOISE)
    
            if c.SWITCH_ASSOCIATION_KNOWN == 1:
                zf,idf,zn, da_table= data_associate_known(x,z,ftag_visible, da_table)
            else:
                zf,idf, zn= data_associate(x,P,z,RE, c.GATE_REJECT, c.GATE_AUGMENT) 

            x,P= update(x,P,zf,RE,idf, c.SWITCH_BATCH_UPDATE) 
            x,P= augment(x,P, zn,RE) 
    
        # offline data store
        data= store_data(data, x, P, xtrue)
    data= finalise_data(data);
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
    #data['state'][0]['P'] = P
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
    return data

def finalise_data(data):
    # offline storage finalisation
    data['path'] = data['path'][:, 0:data['i']]
    data['true'] = data['true'][:, 0:data['i']]
    return data
    
