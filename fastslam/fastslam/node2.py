import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import UInt16MultiArray
import numpy as np
import json
from .fastslam1_sim import fastslam1_sim


class FAST(Node):
    def __init__(self):
        super().__init__('fast')
        self.publisher_true = self.create_publisher(Float64MultiArray, 'true', 10)
        self.publisher_path = self.create_publisher(Float64MultiArray, 'path', 10)
        self.publisher_X = self.create_publisher(Float64MultiArray, 'stateX', 10)
        self.publisher_len = self.create_publisher(UInt16MultiArray, 'stateLen', 10)

    def run(self):
        with open(os.path.join(os.getcwd(), 'src/fastslam/fastslam/file.json'), 'r') as fr:
            env = json.load(fr)

        lm = np.array([[],[]])
        for i in range(len(env["lm"])):
            lm = np.append(lm, [[env["lm"][i][0]], [env["lm"][i][1]]], axis = 1)
            
        wp = np.array([[],[]])
        for i in range(len(env["wp"])):
            wp = np.append(wp, [[env["wp"][i][0]], [env["wp"][i][1]]], axis = 1)
            
        data, xtrue, xpath, lmE = fastslam1_sim(lm, wp, env["x3"])
        return data, xtrue, xpath, lmE

def main(args=None): 
    rclpy.init(args=args)

    fast = FAST()
    msgTrue = Float64MultiArray()
    msgPath = Float64MultiArray()
    msgStateX = Float64MultiArray()
    msgStateLen = UInt16MultiArray()
    msgs = [msgTrue, msgPath, msgStateX, msgStateLen]
    publishers = [fast.publisher_true, fast.publisher_path, fast.publisher_X, fast.publisher_len]

    data, xtrue, xpath, lmE = fast.run()
    arrayT = []
    arrayPath = []
    arrayX = []
    stateLen = []
    for j in range(xtrue.shape[1]):
        for i in range(3):
            arrayT.append(xtrue[i,j])
            arrayPath.append(xpath[i,j])
            arrayX.append(xpath[i,j])
        stateLen.append(lmE[j].shape[1]*2 + 3)
        for k in range(lmE[j].shape[1]):
            arrayX.append(lmE[j][0,k])
            arrayX.append(lmE[j][1,k])

    msgs[0].data = arrayT
    msgs[1].data = arrayPath
    msgs[2].data = arrayX
    msgs[3].data = stateLen
            
    for i in range(len(msgs)):
        publishers[i].publish(msgs[i])
    fast.get_logger().info('Finished!')

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    fast.destroy_node()
    rclpy.shutdown()
    #spin_once()
