#!/usr/bin/env python3

import rospy
import actionlib
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from gazebo_msgs.srv import DeleteModel
from gazebo_msgs.srv import DeleteModelRequest

rospy.init_node('node1')
print("cam")
print("d")
rate = rospy.Rate(20)
move = Twist()
scan=LaserScan()
position = Odometry()
l = []
a = 0
def callback(scan):
        print("a")
        global l
        l = scan.ranges
        #print(l)
        return scan
def position1(position):
        global a
        print("b")
        x = position.pose.pose.position.x
        y = position.pose.pose.position.y
        angle = position.pose.pose.orientation
        [roll,pitch,theta] = euler_from_quaternion([angle.x,angle.y,angle.z,angle.w])
        a = theta
        #print(a)
        return position
rospy.Subscriber('scan',LaserScan,callback)
rospy.Subscriber('odom',Odometry,position1)
pub = rospy.Publisher('cmd_vel', Twist, queue_size=2)
#rospy.wait_for_service('/gazebo/delete_model')
#delete_model_service = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
#delete = DeleteModelRequest()
#delete.model_name = "obstacle"
#result = delete_model_service(delete)
#print(result)
while not rospy.is_shutdown():
        rate = rospy.Rate(10)
        
        #if (l==[]) or (a == 0):
            #continue
        print(l)
        rospy.sleep(10)
        while min(l[1:35]) > 0.5 or min(l[320:359]) > 3.0 or l[0] >0.5:
                move.angular.z = 0.0
                move.linear.x = 0.2
                print("enter")
                pub.publish(move)
        while l[0] < 2.0:
                move.linear.x = 0.0
                pub.publish()
                print(l[0:35])
                while (abs(-1.57-a)>0.1):
                    print("rotate")
                    move.angular.z = 0.2
                    pub.publish(move)
        while min(l[60:120]) < 1.0:
                move.angular.z = 0.0
                move.linear.x = 0.2
                print("move")
                pub.publish(move)
        while l[0] < 3.0 and min(l[100:150])<2.5 and l[90]>3.0:
                move.linear.x = 0.0
                pub.publish()
                while (a<0):
                    move.angular.z = 0.2
                    print("90degree")
                    pub.publish(move)
