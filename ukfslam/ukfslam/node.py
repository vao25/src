import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import UInt16MultiArray
import numpy as np
import math
import json
from .ukfslam_sim import ukfslam_sim
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import PointCloud2, PointField
import struct
import time


class UKF(Node):
    def __init__(self):
        super().__init__('ukf')
        #self.publisher_true = self.create_publisher(Float64MultiArray, 'true', 10)
        #self.publisher_path = self.create_publisher(Float64MultiArray, 'path', 10)
        #self.publisher_X = self.create_publisher(Float64MultiArray, 'stateX', 10)
        #self.publisher_P = self.create_publisher(Float64MultiArray, 'stateP', 10)
        #self.publisher_len = self.create_publisher(UInt16MultiArray, 'stateLen', 10)

        self.true_path_publisher_ = self.create_publisher(Path, 'true_path', 10)
        self.estimated_path_publisher_ = self.create_publisher(Path, 'estimated_path', 10)
        self.landmarks_publisher = self.create_publisher(PointCloud2, 'landmarks', 10)    
        self.lm = None

    def run(self):
        with open(os.path.join(os.getcwd(), 'src/ukfslam/ukfslam/file.json'), 'r') as fr:
            env = json.load(fr)

        lm = np.array([[],[]])
        for i in range(len(env["lm"])):
            lm = np.append(lm, [[env["lm"][i][0]], [env["lm"][i][1]]], axis = 1)
            
        wp = np.array([[],[]])
        for i in range(len(env["wp"])):
            wp = np.append(wp, [[env["wp"][i][0]], [env["wp"][i][1]]], axis = 1)
            
        data = ukfslam_sim(lm, wp, env["x3"])
        self.lm = lm
        return data

def main(args=None): 
    rclpy.init(args=args)

    ukf = UKF()
    msgTrue = Float64MultiArray()
    msgPath = Float64MultiArray()
    msgStateX = Float64MultiArray()
    msgStateP = Float64MultiArray()
    msgStateLen = UInt16MultiArray()
    msgs = [msgTrue, msgPath, msgStateX, msgStateP, msgStateLen]
    #publishers = [ekf.publisher_true, ekf.publisher_path, ekf.publisher_X, ekf.publisher_P, ekf.publisher_len]

    data = ukf.run()

    rmse(data['true'], data['path'])
    epsilonT = trans(data['true'], data['path'])
    epsilonR = rot(data['true'], data['path'])
    print("epsilon =", epsilonT + epsilonR)

    arrayT = []
    arrayPath = []
    arrayX = []
    arrayP = []
    stateLen = []
    for j in range(data['true'].shape[1]):
        for i in range(3):
            arrayT.append(data['true'][i,j])
            arrayPath.append(data['path'][i,j])
        stateLen.append(len(data['state'][j]['x']))
        for i in range(len(data['state'][j]['x'])):
            arrayX.append(data['state'][j]['x'][i][0])
            arrayP.append(data['state'][j]['P'][i])
    msgs[0].data = arrayT
    msgs[1].data = arrayPath
    msgs[2].data = arrayX
    msgs[3].data = arrayP
    msgs[4].data = stateLen
            
    #for i in range(len(msgs)):
        #publishers[i].publish(msgs[i])
    ukf.get_logger().info('Finished!')


    path = Path()
    path.header.frame_id = "map"

    for j in range(data['true'].shape[1]):
        pose = PoseStamped()
        #print(type(data['true'][1,j]))
        pose.pose.position.x = data['true'][0,j]
        pose.pose.position.y = data['true'][1,j]
        pose.pose.position.z = 1.0
        path.poses.append(pose)

    estpath = Path()
    estpath.header.frame_id = "map"

    for j in range(data['true'].shape[1]):
        pose = PoseStamped()
        #print(type(data['true'][1,j]))
        pose.pose.position.x = data['path'][0,j]
        pose.pose.position.y = data['path'][1,j]
        pose.pose.position.z = 1.0
        estpath.poses.append(pose)        

    landmarks = PointCloud2()
    landmarks.header.frame_id = "map"

    px = PointField()
    px.name = "x"
    px.offset = 0
    px.datatype = PointField.FLOAT32
    px.count = 1

    py = PointField()
    py.name = "y"
    py.offset = 4
    py.datatype = PointField.FLOAT32
    py.count = 1

    pz = PointField()
    pz.name = "z"
    pz.offset = 8
    pz.datatype = PointField.FLOAT32
    pz.count = 1

    rgba = PointField()
    rgba.name = "rgba"
    rgba.offset = 12
    rgba.datatype = PointField.UINT32
    rgba.count = 1


    fields = [
        px, py, pz, rgba
    ]
    point_struct = struct.Struct("<fffBBBB")
    points = []
    #print(ekf.lm.shape)
    for j in range(ukf.lm.shape[1]):
        p = Point()
        p.x = ukf.lm[0, j]
        p.y = ukf.lm[1, j]
        p.z = 1.0      
        points.append(p)

    buffer = bytearray(point_struct.size * len(points))
    for i, point in enumerate(points):    
        point_struct.pack_into(
            #buffer, i * point_struct.size, point.x, point.y, point.z, int("0x0000ffff",0)
            buffer, i * point_struct.size, point.x, point.y, point.z, 255,255,255,255
        )

    landmarks.height = 1
    landmarks.width = len(points)
    landmarks.is_dense=False
    landmarks.is_bigendian=False
    landmarks.fields=fields
    landmarks.point_step=point_struct.size
    #print(int(len(buffer)))
    landmarks.row_step=int(len(buffer))
    landmarks.data=buffer

    for i in range(100):
        ukf.true_path_publisher_.publish(path)    
        ukf.estimated_path_publisher_.publish(estpath) 
        ukf.landmarks_publisher.publish(landmarks)           
        time.sleep(1)


    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    ukf.destroy_node()
    rclpy.shutdown()
    #spin_once()


def rmse(xtrue, xpath):
    xtruePos = xtrue[0:2,:]
    xpathPos = xpath[0:2,:]
    diff = xpathPos - xtruePos
    M = diff @ diff.T
    result = math.sqrt((M[0,0] + M[1,1]) / xtrue.shape[1])
    print("RMSE =", result)

def trans(xtrue, xpath):
    deltaTrue = np.zeros((2, xtrue.shape[1]-1))
    deltaPath = np.zeros((2, xtrue.shape[1]-1))
    for i in range(xtrue.shape[1]-1):
        deltaTrue[0,i] = xtrue[0,i] - xtrue[0,i+1]
        deltaTrue[1,i] = xtrue[1,i] - xtrue[1,i+1]
        deltaPath[0,i] = xpath[0,i] - xpath[0,i+1]
        deltaPath[1,i] = xpath[1,i] - xpath[1,i+1]
    delta = deltaPath - deltaTrue
    M = delta @ delta.T
    epsilonTrans = (M[0,0] + M[1,1]) / (xtrue.shape[1]-1)
    print("epsilonTrans =", epsilonTrans)
    return epsilonTrans

def rot(xtrue, xpath):
    deltaTrue = np.zeros(xtrue.shape[1]-1)
    deltaPath = np.zeros(xtrue.shape[1]-1)
    for i in range(xtrue.shape[1]-1):
        deltaTrue[i] = xtrue[2,i] - xtrue[2,i+1]
        deltaPath[i] = xpath[2,i] - xpath[2,i+1]
    delta = deltaPath - deltaTrue
    delta = np.absolute(delta)
    a1 = delta
    a2 = 2*math.pi - delta
    rotation = np.zeros(xtrue.shape[1]-1)
    for i in range(xtrue.shape[1]-1):
        rotation[i] = min(a1[i], a2[i])
    epsilonRot = np.sum(rotation**2) / (xtrue.shape[1]-1)
    print("epsilonRot =", epsilonRot)
    return epsilonRot
 
