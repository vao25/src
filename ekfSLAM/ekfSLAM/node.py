import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
import json
from .ekfslam_sim import ekfslam_sim


class EKF(Node):
    def __init__(self):
        super().__init__('ekf')
        self.publisher_true = self.create_publisher(Float64MultiArray, 'true', 10)
        self.publisher_path = self.create_publisher(Float64MultiArray, 'path', 10)
        self.publisher_X = self.create_publisher(Float64MultiArray, 'stateX', 10)
        self.publisher_P = self.create_publisher(Float64MultiArray, 'stateP', 10)
        self.publisher_len = self.create_publisher(Float64MultiArray, 'stateLen', 10)

def run():
    with open(os.path.join(os.getcwd(), 'src/ekfSLAM/ekfSLAM/file.json'), 'r') as fr:
        env = json.load(fr)

    lm = np.array([[],[]])
    for i in range(len(env["lm"])):
        lm = np.append(lm, [[env["lm"][i][0]], [env["lm"][i][1]]], axis = 1)
        
    wp = np.array([[],[]])
    for i in range(len(env["wp"])):
        wp = np.append(wp, [[env["wp"][i][0]], [env["wp"][i][1]]], axis = 1)
        
    data = ekfslam_sim(lm, wp, env["x3"])
    return data

def main(args=None): 
    rclpy.init(args=args)

    ekf = EKF()
    msgTrue = Float64MultiArray()
    msgPath = Float64MultiArray()
    msgStateX = Float64MultiArray()
    msgStateP = Float64MultiArray()
    msgStateLen = Float64MultiArray()
    msgs = [msgTrue, msgPath, msgStateX, msgStateP, msgStateLen]
    publishers = [ekf.publisher_true, ekf.publisher_path, ekf.publisher_X, ekf.publisher_P, ekf.publisher_len]

    data = run()
    arrayT = []
    arrayPath = []
    arrayX = []
    arrayP = []
    stateLen = []
    for j in range(data['true'].shape[1]):
        for i in range(3):
            arrayT.append(data['true'][i,j])
            arrayPath.append(data['path'][i,j])
        stateLen.append(float(len(data['state'][j]['x'])))
        for i in range(len(data['state'][j]['x'])):
            arrayX.append(data['state'][j]['x'][i][0])
            arrayP.append(data['state'][j]['P'][i])
    msgs[0].data = arrayT
    msgs[1].data = arrayPath
    msgs[2].data = arrayX
    msgs[3].data = arrayP
    msgs[4].data = stateLen
            
    for i in range(len(msgs)):
        publishers[i].publish(msgs[i])
    ekf.get_logger().info('Finished!')

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    ekf.destroy_node()
    rclpy.shutdown()
    #spin_once()
    
