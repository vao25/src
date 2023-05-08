import numpy as np
from . import configfile as c # ** USE THIS FILE TO CONFIGURE THE Fast-SLAM **
from .compute_steering import compute_steering
from .predict_true import predict_true
from .add_control_noise import add_control_noise
from .predict import predict
from .get_observations import get_observations
from .add_observation_noise import add_observation_noise
from .data_associate_known import data_associate_known
from .compute_weight import compute_weight
from .feature_update import feature_update
from .add_feature import add_feature
from .resample_particles import resample_particles


def fastslam1_sim(lm, wp, phi):
    """
    data= fastslam1_sim(lm, wp, phi)

     INPUTS: 
       lm - set of landmarks
       wp - set of waypoints
       phi - initial rotation angle


     OUTPUTS:
       data - set of particles representing final state

     NOTES:
       This program is a FastSLAM 1.0 simulator. To use, create a set of landmarks and 
       vehicle waypoints (ie, waypoints for the desired vehicle path). The program
       'setup.py' may be used to create this simulated environment.
           The configuration of the simulator is managed by the script file
       'configfile.py'. To alter the parameters of the vehicle, sensors, etc
       adjust this file. There are also several switches that control certain
       filter options.
    """

    if c.SWITCH_PREDICT_NOISE == 0:
        print('Sampling from predict noise is necessary for FastSLAM 1.0 particle diversity')
    
    # initialisations
    particles = initialise_particles(c.NPARTICLES, phi)
    xtrue = np.zeros((3, 1))
    xtrue[2] = phi
    xpath = np.zeros((3, 1))
    xpath[2] = phi
    lms = [np.array([[],[]])]
    
    dt = c.DT_CONTROLS # change in time between predicts
    dtsum = 0 # change in time since last observation
    ftag = np.arange(lm.shape[1]) # identifier for each landmark
    da_table = np.zeros((1, lm.shape[1])) - 1 # data association table 
    iwp = 0 # index to first waypoint 
    G = 0 # initial steer angle
    
    if c.SWITCH_SEED_RANDOM != 0:
        np.random.seed(c.SWITCH_SEED_RANDOM)
    
    Qe = np.copy(c.Q)
    Re = np.copy(c.R)
    if c.SWITCH_INFLATE_NOISE == 1:
        Qe = 2 * np.copy(c.Q)
        Re = 2 * np.copy(c.R)
    
    NUMBER_LOOPS = c.NUMBER_LOOPS
    
    # Main loop 
    while iwp != -1:
        
        # Compute true data
        G, iwp = compute_steering(xtrue[:,[-1]], wp, iwp, c.AT_WAYPOINT, G, c.RATEG, c.MAXG, dt)
        if iwp == -1 and NUMBER_LOOPS > 1:
            iwp = 0
            NUMBER_LOOPS = NUMBER_LOOPS - 1 # path loopfs repeat
        xtrueI = predict_true(xtrue[:,[-1]], c.V, G, c.WHEELBASE, dt)
        xtrue = np.hstack((xtrue,xtrueI))
        
        # Add process noise
        Vn, Gn = add_control_noise(c.V, G, c.Q, c.SWITCH_CONTROL_NOISE)
        
        # Predict step
        for i in range(c.NPARTICLES):
            particles[i] = predict(particles[i], Vn, Gn, Qe, c.WHEELBASE, dt, c.SWITCH_PREDICT_NOISE)
            if c.SWITCH_HEADING_KNOWN == 1:
                particles[i].xv[2] = xtrue[2,-1]
        
        # Observe step
        dtsum = dtsum + dt
        if dtsum >= c.DT_OBSERVE:
            dtsum = 0
            
            # Compute true data, then add noise
            z, ftag_visible = get_observations(xtrue[:,[-1]], lm, ftag, c.MAX_RANGE)
            z = add_observation_noise(z, c.R, c.SWITCH_SENSOR_NOISE)
            
            # Compute (known) data associations
            Nf = particles[0].xf.shape[1]
            zf, idf, zn, da_table = data_associate_known(z, ftag_visible, da_table, Nf)
            
            # Perform update
            for i in range(c.NPARTICLES):
                if len(zf) != 0: # observe map features
                    w = compute_weight(particles[i], zf, idf, c.R) # w = p(z_k | x_k)
                    particles[i].w = particles[i].w * w
                    particles[i] = feature_update(particles[i], zf, idf, c.R)
                
                if len(zn) != 0: # observe new features, augment map
                    particles[i] = add_feature(particles[i], zn, c.R)
            
            particles = resample_particles(particles, c.NEFFECTIVE, c.SWITCH_RESAMPLE)
        
        wmax = 0
        j = 0
        for i in range(c.NPARTICLES):
            if particles[i].w > wmax:
                wmax = particles[i].w
                j = i
        xpath = np.hstack((xpath,particles[j].xv))     
        lms.append(np.copy(particles[j].xf))
    
    data = particles
    return data, xtrue, xpath, lms

class Particle:
    def __init__(self, w, phi):
        self.w = w
        self.xv = np.array([[0],[0],[phi]])
        self.xf = np.array([[],[]])
        self.Pf = np.array([ [[],[]], [[],[]] ])


def initialise_particles(np, phi):
    w = 1/np
    p = []
    for i in range(np):
        p.append(Particle(w,phi))
    return p
