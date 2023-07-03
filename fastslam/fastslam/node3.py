import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import UInt16MultiArray
import numpy as np
import json
from .fastslam1_sim import fastslam1_sim
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import PointCloud2, PointField
import struct
import time


class FAST(Node):
    def __init__(self):
        super().__init__('fast')
        #self.publisher_true = self.create_publisher(Float64MultiArray, 'true', 10)
        #self.publisher_path = self.create_publisher(Float64MultiArray, 'path', 10)
        #self.publisher_X = self.create_publisher(Float64MultiArray, 'stateX', 10)
        #self.publisher_len = self.create_publisher(UInt16MultiArray, 'stateLen', 10)

        self.true_path_publisher_ = self.create_publisher(Path, 'true_path', 10)
        self.estimated_path_publisher_ = self.create_publisher(Path, 'estimated_path', 10)
        self.landmarks_publisher = self.create_publisher(PointCloud2, 'landmarks', 10)
        self.publisher_ = self.create_publisher(Float64MultiArray, 'estimated_path', 10)      
        self.lm = None

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
        self.lm = lm
        return data, xtrue, xpath, lmE

def main(args=None): 
    rclpy.init(args=args)

    fast = FAST()
    msgTrue = Float64MultiArray()
    msgPath = Float64MultiArray()
    msgStateX = Float64MultiArray()
    msgStateLen = UInt16MultiArray()
    msgs = [msgTrue, msgPath, msgStateX, msgStateLen]
    #publishers = [fast.publisher_true, fast.publisher_path, fast.publisher_X, fast.publisher_len]

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
            
    #for i in range(len(msgs)):
        #publishers[i].publish(msgs[i])
    fast.get_logger().info('Finished!')


    path = Path()
    path.header.frame_id = "map"

    for j in range(xtrue.shape[1]):
        pose = PoseStamped()
        #print(type(data['true'][1,j]))
        pose.pose.position.x = xtrue[0,j]
        pose.pose.position.y = xtrue[1,j]
        pose.pose.position.z = 1.0
        path.poses.append(pose)

    estpath = Path()
    estpath.header.frame_id = "map"

    for j in range(xtrue.shape[1]):
        pose = PoseStamped()
        #print(type(data['true'][1,j]))
        pose.pose.position.x = xpath[0,j]
        pose.pose.position.y = xpath[1,j]
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
    #print(fast.lm.shape)
    for j in range(fast.lm.shape[1]):
        p = Point()
        p.x = fast.lm[0, j]
        p.y = fast.lm[1, j]
        p.z = 1.0      
        points.append(p)

    buffer = bytearray(point_struct.size * len(points))
    for i, point in enumerate(points):    
        point_struct.pack_into(
            #buffer, i * point_struct.size, point.x, point.y, point.z, int("0x0000ffff",0)
            buffer, i * point_struct.size, point.x, point.y, point.z, 255,0,0,255
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
        fast.true_path_publisher_.publish(path)    
        fast.estimated_path_publisher_.publish(estpath) 
        fast.landmarks_publisher.publish(landmarks)           
        time.sleep(1)


    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    fast.destroy_node()
    rclpy.shutdown()
    #spin_once()
