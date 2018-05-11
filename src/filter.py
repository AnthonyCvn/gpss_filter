#!/usr/bin/env python
import rospy
import tf
import message_filters
import numpy as np
from scipy import linalg

from ar_track_alvar_msgs.msg import AlvarMarkers
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from math import sin, cos, pi, asin, atan2, sqrt

# Enable/Disable debug information
DEBUG0 = False
DEBUG1 = False
DEBUG2 = False


class SensorFilter:
    def __init__(self):
        # Camera0 to world transform
        self.cam0_to_world = Twist()
        self.cam0_to_world.linear.x = 0.0
        self.cam0_to_world.linear.y = 0.0
        self.cam0_to_world.linear.z = 2.51
        self.cam0_to_world.angular.x = 0.0
        self.cam0_to_world.angular.y = 0.0
        self.cam0_to_world.angular.z = 0.0

        # Control action publisher
        self.pub_cmd = rospy.Publisher('/robot0/diff_drive_controller/cmd_vel', Twist, queue_size=1)

        # Pose estimation publisher
        self.pub_pose = rospy.Publisher('/robot0/pose_estimate', Twist, queue_size=1)

        # Marker position relative to world coordinate publisher
        self.pub_global_marker_position = rospy.Publisher('/robot0/global_marker_position', Twist, queue_size=1)

        # Robot odom position relative to world coordinate publisher
        self.pub_global_odom_position = rospy.Publisher('/robot0/global_odom_position', Twist, queue_size=1)

        # TF broadcaster
        self.br = tf.TransformBroadcaster()

        # TF transform
        self.T_world2cam = tf.transformations.euler_matrix(0, pi, pi / 2, 'sxyz')
        self.T_world2cam[2, 3] = 2.50  # Adding z translation
        self.T_odom_world2robot = tf.transformations.euler_matrix(0, 0, 0, 'sxyz')
        self.T_world2robot = tf.transformations.euler_matrix(0, 0, 0, 'sxyz')
        self.T_world2odom = tf.transformations.euler_matrix(0, 0, 0, 'sxyz')

        # ROS message
        self.cmd_vel = Twist()
        self.pose_estimate = Twist()
        self.global_marker_position = Twist()
        self.global_odom_position = Twist()
        self.last_odom = Odometry()

        # Sensors values storage
        self.global_odom = Twist()
        self.global_marker = Twist()

        # Initialization and Time variables
        self.initialize = True
        self.last_time = 0
        self.real_time = 0
        self.dt = 0

        # Dimension setting
        self.dim_x = 3
        self.dim_u = 2
        self.dim_z = 6

        # Model variable
        self.x = np.zeros((self.dim_x, 1))
        self.u = np.zeros((self.dim_u, 1))
        self.z = np.zeros((self.dim_z, 1))
        self.mu = np.zeros((self.dim_x, 1))
        self.last_mu = np.zeros((self.dim_x, 1))

        # Extended Kalman Filter matrices
        self.S = 0.1 * np.eye(self.dim_x)
        self.H = np.vstack((np.eye(self.dim_x), np.eye(self.dim_x)))

        # Covariance matrices (Q: measurement, R: state transition)
        self.Q = 0.01 * np.eye(self.dim_z)
        self.R = 0.01 * np.eye(self.dim_x)

    def sensors_fusion(self, odom, marker):

        if not marker.markers:
            if DEBUG1:
                print "No marker detected"
            return

        # Timestamp estimation based on odometry
        self.real_time = (odom.header.stamp.nsecs * 1.0) * 1e-9 + odom.header.stamp.secs

        # Initialization on first triggered data
        if self.initialize:
            self.initialize = False
            self.last_time = self.real_time
            if DEBUG0:
                rospy.loginfo("Filter initialized.")
            return

        # Time difference and old time storage.
        self.dt = self.real_time - self.last_time

        # Odometry and Marker measurement.
        quaternion_odom2robot = (
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
            odom.pose.pose.orientation.w)
        T_odom2robot = tf.transformations.quaternion_matrix(quaternion_odom2robot)
        T_odom2robot[0, 3] = odom.pose.pose.position.x
        T_odom2robot[1, 3] = odom.pose.pose.position.y
        T_odom2robot[2, 3] = odom.pose.pose.position.z

        self.T_odom_world2robot = self.T_world2odom.dot(T_odom2robot)

        self.z[0] = self.T_odom_world2robot[0, 3]
        self.z[1] = self.T_odom_world2robot[1, 3]
        self.z[2] = atan2(self.T_odom_world2robot[1, 0], self.T_odom_world2robot[0, 0])

        quaternion_cam2marker = (
            marker.markers[0].pose.pose.orientation.x,
            marker.markers[0].pose.pose.orientation.y,
            marker.markers[0].pose.pose.orientation.z,
            marker.markers[0].pose.pose.orientation.w)

        T_cam2marker = tf.transformations.quaternion_matrix(quaternion_cam2marker)
        T_cam2marker[0, 3] = marker.markers[0].pose.pose.position.x
        T_cam2marker[1, 3] = marker.markers[0].pose.pose.position.y
        T_cam2marker[2, 3] = marker.markers[0].pose.pose.position.z

        T_world2marker = self.T_world2cam.dot(T_cam2marker)

        self.z[3] = T_world2marker[0, 3]
        self.z[4] = T_world2marker[1, 3]
        self.z[5] = atan2(T_world2marker[1, 0], T_world2marker[0, 0])

        # Extended Kalman Filter
        self.ekf()

        # Robot's pose publisher.
        self.pose_estimate.linear.x = self.mu[0]
        self.pose_estimate.linear.y = self.mu[1]
        self.pose_estimate.angular.z = self.mu[2]
        self.pub_pose.publish(self.pose_estimate)

        # Marker's global-position publisher.
        self.global_marker_position.linear.x = T_world2marker[0, 3]
        self.global_marker_position.linear.y = T_world2marker[1, 3]
        self.global_marker_position.linear.z = T_world2marker[2, 3]
        self.pub_global_marker_position.publish(self.global_marker_position)

        # Odom's global-position publisher.
        self.global_odom_position.linear.x = self.T_odom_world2robot[0, 3]
        self.global_odom_position.linear.y = self.T_odom_world2robot[1, 3]
        self.global_odom_position.linear.z = self.T_odom_world2robot[2, 3]
        self.pub_global_odom_position.publish(self.global_odom_position)

        # Update robot pose
        self.T_world2robot = tf.transformations.euler_matrix(0, 0, self.mu[2], 'sxyz')
        self.T_world2robot[0, 3] = self.mu[0]
        self.T_world2robot[1, 3] = self.mu[1]
        self.T_world2odom = self.T_world2robot.dot(linalg.inv(T_odom2robot))

        quaternion_world2odom = tf.transformations.quaternion_from_matrix(self.T_world2odom)
        self.br.sendTransform(self.T_world2odom[0:3, 3],
                              quaternion_world2odom,
                              rospy.Time.now(),
                              "robot0/odom",
                              "/world")
        # Controller
        self.cmd_vel.angular.z = 0.15
        self.cmd_vel.linear.x = 0.1
        self.pub_cmd.publish(self.cmd_vel)

        # Storage
        self.last_odom = odom
        self.last_time = self.real_time
        self.u[0] = self.cmd_vel.linear.x
        self.u[1] = self.cmd_vel.angular.z

        # Log information
        if DEBUG0:
            rospy.loginfo("Time synchronization error (ms): {0}".format(
                str(abs(odom.header.stamp.nsecs - marker.header.stamp.nsecs) * 1e-6)))
            rospy.loginfo("Sampling frequency (Hz): " + str(1 / self.dt))

        if DEBUG1:
            rospy.loginfo(
                "X estimate = {0}Y estimate: {1}Orientation estimate: {2}".format(str(self.mu[0]),
                                                                                  str(self.mu[1]),
                                                                                  str(self.mu[2] / pi * 180)))

        if DEBUG2:
            rospy.loginfo("x_o" + str(self.z[0]) + ", y_o" + str(self.z[1]) + ", Theta_o" + str(self.z[2] / pi * 180))
            rospy.loginfo("x_m" + str(self.z[3]) + ", y_m" + str(self.z[4]) + ", Theta_m" + str(self.z[5] / pi * 180))

    def prediction_model(self, u, x, dt):
        g = np.zeros((self.dim_x, 1))
        g[0] = x[0] + dt * u[0] * cos(x[2])
        g[1] = x[1] - dt * u[0] * sin(x[2])
        g[2] = x[2] + dt * u[1]
        return g

    @staticmethod
    def observation_model(x):
        return np.vstack((x, x))

    def jacobian_derivation(self, u, x, dt):
        G = np.eye(self.dim_x)
        G[0, 2] = - dt * u[0] * sin(x[2])
        G[1, 2] = - dt * u[0] * cos(x[2])
        return G

    def ekf(self):
        # Update G
        self.G = self.jacobian_derivation(self.u, self.mu, self.dt)

        # Prediction
        self.mu = self.prediction_model(self.u, self.mu, self.dt)
        self.S = self.G.dot(self.S).dot(self.G.T) + self.R

        # Optimal Kalman gain
        optimal_gain = self.S.dot(self.H.T).dot(linalg.inv(self.H.dot(self.S).dot(self.H.T) + self.Q))

        # Measurement update
        self.mu = self.mu + optimal_gain.dot(self.z - self.observation_model(self.mu))
        self.S = (np.eye(self.dim_x) - optimal_gain.dot(self.H)).dot(self.S)


def main():
    # Init gpss_filter node.
    rospy.init_node("gpss_filter")

    # Add filter object
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
