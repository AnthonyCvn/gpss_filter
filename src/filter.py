#!/usr/bin/env python
import rospy
import message_filters
import numpy as np
from scipy import linalg

from ar_track_alvar_msgs.msg import AlvarMarkers
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from math import sin, cos

# Enable/Disable debug information
DEBUG = True


class SensorFilter:

    def __init__(self):
        # Control action publisher
        self.pub_cmd = rospy.Publisher('/robot0/diff_drive_controller/cmd_vel', Twist, queue_size=1)

        # Pose estimation publisher
        self.pub_pose = rospy.Publisher('/robot0/pose_estimate', Twist, queue_size=1)

        # ROS message
        self.cmd_vel = Twist()
        self.pose_estimate = Twist()

        # Initialization and Time variables
        self.initialize = True
        self.last_time = 0
        self.real_time = 0
        self.dt = 0

        # Dimension setting
        self.dim_x = 3
        self.dim_u = 2
        self.dim_z = 6

        # Kinematic model
        self.g = np.zeros((self.dim_x, 1))

        # Model variable
        self.x = np.zeros((self.dim_x, 1))
        self.u = np.zeros((self.dim_u, 1))
        self.z = np.zeros((self.dim_z, 1))
        self.mu = np.zeros((self.dim_x, 1))
        self.last_mu = np.zeros((self.dim_x, 1))

        # Extended Kalman Filter matrices
        self.S = np.zeros((self.dim_x, self.dim_x))
        self.last_S = np.zeros((self.dim_x, self.dim_x))
        self.G = np.zeros((self.dim_x, self.dim_x))
        self.H = np.zeros((self.dim_z, self.dim_x))

        # Covariance matrices (Q: measurement, R: state transition)
        self.Q = np.eye(self.dim_z)
        self.R = np.eye(self.dim_x)

    def sensors_fusion(self, odom, marker):

        # Timestamp estimation based on odometry
        self.real_time = (odom.header.stamp.nsecs*1.0)*1e-9 + odom.header.stamp.secs

        # Initialization on first triggered data
        if self.initialize:
            self.initialize = False
            self.last_time = self.real_time
            if DEBUG:
                rospy.loginfo("Filter initialized.")
            return

        # Time difference and old time storage
        self.dt = self.real_time - self.last_time
        self.last_time = self.real_time

        # Parameters update
        self.u[0] = self.cmd_vel.linear.x
        self.u[1] = self.cmd_vel.angular.z

        # Extended Kalman Filter
        self.ekf()

        # Publish the estimation of the robot's pose
        self.pose_estimate.linear.x = self.mu[0]
        self.pose_estimate.linear.y = self.mu[1]
        self.pose_estimate.angular.z = self.mu[3]
        self.pub_pose.publish(self.pose_estimate)

        # Controller
        # self.cmd_vel.angular.z = 0.1
        # self.cmd_vel.linear.x = 0.1
        self.pub_cmd.publish(self.cmd_vel)

        # Log information
        if DEBUG:
            rospy.loginfo("Time synchronization error (ms): "
                          + str(abs(odom.header.stamp.nsecs - marker.header.stamp.nsecs)*1e-6))
            rospy.loginfo("Sampling frequency (Hz): " + str(1/self.dt))

    def prediction_model(self, u, x, dt):
        if u[1] > 0.001:
            self.g[0] = x[0] - u[0]*1.0/u[1] * (1 - sin(x[2]+u[1]*dt))
            self.g[1] = x[1] + u[0]*1.0/u[1] * (1 - cos(x[2]+u[1]*dt))
            self.g[2] = x[2] + u[1] * dt
        else:
            self.g[0] = x[0] + u[0] * cos(x[2])
            self.g[1] = x[1] + u[0] * sin(x[2])
            self.g[2] = x[2]
        return self.g

    @staticmethod
    def observation_model(x):
        return np.vstack((x, x))

    def ekf(self):
        # Prediction
        self.mu = self.prediction_model(self.u, self.last_mu, self.dt)
        self.S = self.G.dot(self.last_S).dot(self.G.T) + self.R

        # Optimal Kalman gain
        optimal_gain = self.S.dot(self.H.T).dot(linalg.inv(self.H.dot(self.S).dot(self.H.T) + self.Q))

        # Measurement update
        self.mu = self.mu + optimal_gain.dot(self.z - self.observation_model(self.mu))
        self.S = (np.eye(self.dim_x) - optimal_gain.dot(self.H)).dot(self.S)


def main():
    # Init gpss_filter node.
    rospy.init_node("gpss_filter")

    # Add sensor filter object
    sf = SensorFilter()

    # Subscribe to Odometry and ALVAR tag topics.
    odom_sub = message_filters.Subscriber("/robot0/diff_drive_controller/odom", Odometry)
    marker_sub = message_filters.Subscriber("/ar_pose_marker", AlvarMarkers)

    # Synchronize topics with message_filter.
    ts = message_filters.ApproximateTimeSynchronizer([odom_sub, marker_sub], 1, 0.010, allow_headerless=False)
    ts.registerCallback(sf.sensors_fusion)

    # Blocks until ROS node is shutdown.
    rospy.spin()

if __name__ == "__main__":
    main()