#!/usr/bin/env python
import rospy
import tf
import message_filters
import numpy as np
import time
from scipy import linalg

from ar_track_alvar_msgs.msg import AlvarMarkers
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Path
from nav_msgs.srv import GetMap
from geometry_msgs.msg import PoseStamped

from math import sin, cos, pi, atan2, sqrt, fabs

import matplotlib.pyplot as plt
import a_star
import reeds_shepp

# Enable/Disable
DEBUG = False
controller_debug = False
show_animation = True
is_a_star = False
is_reeds_sheep = True


class TicToc:
    """
    Time measurement object
    t = TicToc()
    t.tic()
    time.sleep(2)
    print t.toc()
    2.00210309029
    """
    def __init__(self):
        self.stack = []
        self. named = {}
        self.elapsed = 0

    def tic(self, name=None):
        if name is None:
            self.stack.append(time.time())
        else:
            self.named[name] = time.time()

    def toc(self, name=None):
        if name is None:
            start = self.stack.pop()
        else:
            start = self.named.pop(name)
        self.elapsed = time.time() - start
        return self.elapsed

    def set(self, value, name=None):
        if name is None:
            self.stack[-1] = value
        else:
            self.named[name] = value

    def get(self, name=None):
        if name is None:
            return self.stack[-1]
        else:
            return self.named[name]


class Robot:
    def __init__(self):
        # Ros Publisher
        self.pub_global_odom_position = rospy.Publisher('/robot0/global_odom_position', Twist, queue_size=1)
        self.pub_global_model_predicted_pose = rospy.Publisher('/robot0/global_model_predicted_position', Twist, queue_size=1)
        self.pub_global_marker_position = rospy.Publisher('/robot0/global_marker_position', Twist, queue_size=1)
        self.pub_pose = rospy.Publisher('/robot0/pose_estimate', Twist, queue_size=1)

        # TF broadcaster
        self.br = tf.TransformBroadcaster()

        # ROS variables
        self.global_model_predicted_pose = Twist()
        self.global_odom_position = Twist()
        self.global_marker_position = Twist()
        self.pose = Twist()

        # Matrix transforms
        self.T_world2cam = tf.transformations.euler_matrix(0, pi, pi / 2, 'sxyz')
        self.T_world2cam[2, 3] = 2.50  # Adding z translation
        self.T_odom_world2robot = tf.transformations.euler_matrix(0, 0, 0, 'sxyz')
        self.T_world2robot = tf.transformations.euler_matrix(0, 0, 0, 'sxyz')
        self.T_world2odom = tf.transformations.euler_matrix(0, 0, 0, 'sxyz')
        self.T_odom2robot = tf.transformations.euler_matrix(0, 0, 0, 'sxyz')
        self.T_world2marker = tf.transformations.euler_matrix(0, 0, 0, 'sxyz')

        # sensors pose
        self.odom_w2r = np.zeros((3, 1))
        self.marker_w2r = np.zeros((3, 1))

        # Controller of the robot
        self.controller = Controller()

    def global_odom(self, odom):
        quaternion_odom2robot = (
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
            odom.pose.pose.orientation.w)
        self.T_odom2robot = tf.transformations.quaternion_matrix(quaternion_odom2robot)
        self.T_odom2robot[0, 3] = odom.pose.pose.position.x
        self.T_odom2robot[1, 3] = odom.pose.pose.position.y
        self.T_odom2robot[2, 3] = odom.pose.pose.position.z

        self.T_odom_world2robot = self.T_world2odom.dot(self.T_odom2robot)

        self.odom_w2r[0] = self.T_odom_world2robot[0, 3]
        self.odom_w2r[1] = self.T_odom_world2robot[1, 3]
        self.odom_w2r[2] = atan2(self.T_odom_world2robot[1, 0], self.T_odom_world2robot[0, 0])

        return self.odom_w2r

    def global_odom_pub(self):
        self.global_odom_position.linear.x = self.T_odom_world2robot[0, 3]
        self.global_odom_position.linear.y = self.T_odom_world2robot[1, 3]
        self.global_odom_position.angular.z = atan2(self.T_odom_world2robot[1, 0], self.T_odom_world2robot[0, 0])
        self.pub_global_odom_position.publish(self.global_odom_position)

    def global_marker(self, marker):
        quaternion_cam2marker = (
            marker.markers[0].pose.pose.orientation.x,
            marker.markers[0].pose.pose.orientation.y,
            marker.markers[0].pose.pose.orientation.z,
            marker.markers[0].pose.pose.orientation.w)

        T_cam2marker = tf.transformations.quaternion_matrix(quaternion_cam2marker)
        T_cam2marker[0, 3] = marker.markers[0].pose.pose.position.x
        T_cam2marker[1, 3] = marker.markers[0].pose.pose.position.y
        T_cam2marker[2, 3] = marker.markers[0].pose.pose.position.z

        self.T_world2marker = self.T_world2cam.dot(T_cam2marker)

        self.marker_w2r[0] = self.T_world2marker[0, 3]
        self.marker_w2r[1] = self.T_world2marker[1, 3]
        self.marker_w2r[2] = atan2(self.T_world2marker[1, 0], self.T_world2marker[0, 0])

        return self.marker_w2r

    def global_marker_pub(self, pose):
        self.global_marker_position.linear.x = pose[0]
        self.global_marker_position.linear.y = pose[1]
        self.global_marker_position.angular.z = pose[2]
        self.pub_global_marker_position.publish(self.global_marker_position)

    def world2odom(self, robot_pose):

        self.T_world2robot = tf.transformations.euler_matrix(0, 0, robot_pose[2], 'sxyz')
        self.T_world2robot[0, 3] = robot_pose[0]
        self.T_world2robot[1, 3] = robot_pose[1]
        self.T_world2odom = self.T_world2robot.dot(linalg.inv(self.T_odom2robot))

        quaternion_world2odom = tf.transformations.quaternion_from_matrix(self.T_world2odom)
        self.br.sendTransform(self.T_world2odom[0:3, 3],
                              quaternion_world2odom,
                              rospy.Time.now(),
                              "robot0/odom",
                              "/world")

    def update_pose(self, pose, publish=True):
        self.pose.linear.x = pose[0]
        self.pose.linear.y = pose[1]
        self.pose.angular.z = pose[2]
        if publish:
            self.pub_pose.publish(self.pose)

    def update_prediction(self, prediction, publish=True):
        self.global_model_predicted_pose.linear.x = prediction[0]
        self.global_model_predicted_pose.linear.y = prediction[1]
        self.global_model_predicted_pose.angular.z = prediction[2]
        if publish:
            self.pub_global_model_predicted_pose.publish(self.global_model_predicted_pose)


class SensorsFilter:
    def __init__(self):
        # Initialization and Time variables
        self.initialize = True
        self.calibrate = True
        self.dt = 0
        self.t = TicToc()

        # Robot object.
        self.robot = Robot()

        # Model variable
        self.dim_x = 3
        self.dim_u = 2
        self.dim_z = 6
        self.x = np.zeros((self.dim_x, 1))
        self.u = np.zeros((self.dim_u, 1))
        self.z = np.zeros((self.dim_z, 1))
        self.z_storage = np.zeros((self.dim_z, 1))
        self.mu = np.zeros((self.dim_x, 1))
        self.mu_storage = self.mu

        # Variance and measurement matrix.
        self.S = 0.1 * np.eye(self.dim_x)
        self.H = np.vstack((np.eye(self.dim_x), np.eye(self.dim_x)))

        # Covariance matrices (Q: measurement (odometry then markers), R: state transition).
        self.Q = np.diag(np.array([1.0e-3, 1.0e-3, 1.0e-3, 5.0e-3, 5.0e-3, 5.0e-3]))
        self.R = 1.0e-3 * np.eye(self.dim_x)

        # Motion model threshold
        self.epsilon = 1.0e-3

    def fusion(self, odom, marker):
        self.t.tic(1)

        # Initialization on first triggered data
        if self.initialize:
            self.t.set(self.t.get(1), 0)
            if not marker.markers:
                if DEBUG:
                    rospy.loginfo("No marker detected")
                return
            else:
                self.initialize = False
                rospy.loginfo("Filter initialized.")

        # Time difference and time storage.
        self.dt = self.t.get(1) - self.t.get(0)
        self.t.set(self.t.get(1), 0)

        # Odometry and Marker measurement.
        self.z[0:3] = self.robot.global_odom(odom)

        if not marker.markers:
            if self.calibrate:
                return
            if DEBUG:
                rospy.loginfo("No marker detected")
            self.H[3, 0] = 0
            self.H[4, 1] = 0
            self.H[5, 2] = 0
        else:
            self.H[3, 0] = 1
            self.H[4, 1] = 1
            self.H[5, 2] = 1

            self.z[3:6] = self.robot.global_marker(marker)

            # Marker's global-position publisher.
            self.robot.global_marker_pub(self.z[3:6])

            if self.calibrate:
                self.calibrate = False
                self.mu[0] = self.z[3]
                self.mu[1] = self.z[4]
                self.mu[2] = self.z[5]
                self.z[0] = self.z[3]
                self.z[1] = self.z[4]
                self.z[2] = self.z[5]

        # Extended Kalman Filter (EKF)
        self.ekf()

        self.robot.update_pose(self.mu, True)
        self.robot.update_prediction(self.mu_storage, True)

        # Update TF transform
        self.robot.world2odom(self.mu)

        # Controller
        self.u = self.robot.controller.ctrl(self.robot)

        # Storage
        self.z_storage = self.z

        # Log information
        if DEBUG:
            rospy.loginfo("Time synchronization error (ms): {0}"
                          .format(str(abs(odom.header.stamp.nsecs - marker.header.stamp.nsecs) * 1e-6)))
            rospy.loginfo("Sampling frequency (Hz): " + str(1 / self.dt))
            rospy.loginfo(
                "X estimate = {0}Y estimate: {1} Orientation estimate: {2}".format(str(self.mu[0]),
                                                                                   str(self.mu[1]),
                                                                                   str(self.mu[2] / pi * 180)))
            rospy.loginfo("x_o" + str(self.z[0]) + ", y_o" + str(self.z[1]) + ", Theta_o" + str(self.z[2] / pi * 180))
            rospy.loginfo("x_m" + str(self.z[3]) + ", y_m" + str(self.z[4]) + ", Theta_m" + str(self.z[5] / pi * 180))

    def measure_subtraction(self, z1, z2):
        delta = z1 - z2
        delta[2] = self.wraptopi(delta[2])
        delta[5] = self.wraptopi(delta[5])
        return delta

    @staticmethod
    def wraptopi(angle):
        angle = (angle + pi) % (2 * pi) - pi
        return angle

    def prediction_model(self, u, x, dt):
        g = np.zeros((self.dim_x, 1))

        if fabs(u[1]) < self.epsilon:
            g[0] = x[0] + dt * u[0] * 1.0 * cos(x[2] + u[1] * dt)
            g[1] = x[1] + dt * u[0] * 1.0 * sin(x[2] + u[1] * dt)
        else:
            g[0] = x[0] + u[0] * 1.0 / u[1] * (sin(x[2] + u[1] * dt) - sin(x[2]))
            g[1] = x[1] + u[0] * 1.0 / u[1] * (cos(x[2]) - cos(x[2] + u[1] * dt))
        g[2] = self.wraptopi(x[2] + dt * u[1])
        return g

    def jacobian_prediction_model(self, u, x, dt):
        G = np.eye(self.dim_x)
        if fabs(u[1]) < self.epsilon:
            G[0, 2] = - dt * u[0] * 1.0 * sin(x[2] + u[1] * dt)
            G[1, 2] = + dt * u[0] * 1.0 * cos(x[2] + u[1] * dt)
        else:
            G[0, 2] = u[0] * 1.0 / u[1] * (cos(x[2] + u[1] * dt) - cos(x[2]))
            G[1, 2] = u[0] * 1.0 / u[1] * (sin(x[2] + u[1] * dt) - sin(x[2]))
        return G

    def ekf(self):
        # Update G
        self.G = self.jacobian_prediction_model(self.u, self.mu, self.dt)

        # Prediction
        self.mu = self.prediction_model(self.u, self.mu, self.dt)
        self.mu_storage = self.mu
        self.S = self.G.dot(self.S).dot(self.G.T) + self.R

        # Optimal Kalman gain
        optimal_gain = self.S.dot(self.H.T).dot(linalg.inv(self.H.dot(self.S).dot(self.H.T) + self.Q))

        # Measurement update
        self.mu = self.mu + optimal_gain.dot(self.measure_subtraction(self.z, np.vstack((self.mu, self.mu))))
        self.S = (np.eye(self.dim_x) - optimal_gain.dot(self.H)).dot(self.S)


class Controller:
    def __init__(self):
        # Controller variables
        self.u = np.zeros((2, 1))
        self.i = 0
        self.forward = True

        # Path
        self.path = Path()
        self.goal_i = 0
        self.i_max = 0
        self.seq_change = []
        self.seq_is_reverse = []
        self.idle = True
        self.change_subgoal = True
        self.is_reverse = False

        # ROS variable
        self.goal_pose = Twist()
        self.cmd_vel = Twist()

        # Command publisher
        self.pub_cmd = rospy.Publisher('/robot0/diff_drive_controller/cmd_vel', Twist, queue_size=1)

    def ctrl(self, robot):
        ## Parameters
        # Lyapunnov stability for: k_rho > 0 ; k_alpha > k_rho ; k_beta < 0
        k_rho = 0.7 / pi
        k_alpha = 1.0 / pi
        k_beta = - 1.0 / pi
        dx = 0.0
        rho = 0.0
        dy = 0.0
        phi = 0.0

        # print "change subgoal: " + str(self.change_subgoal)
        # print self.seq_change

        ## Define next goal
        if not self.idle:
            if self.seq_change and self.change_subgoal:
                self.i_max = self.seq_change.pop(0)
                self.change_subgoal = False

            for i in range(self.goal_i, self.i_max):
                dx = self.path.poses[i].pose.position.x - robot.pose.linear.x
                dy = self.path.poses[i].pose.position.y - robot.pose.linear.y
                rho = sqrt(dx ** 2 + dy ** 2)
                if rho >= 0.4:
                    break

            self.goal_i = i
            self.is_reverse = self.seq_is_reverse[i]

            # print "i max: " + str(self.i_max)
            # print "goal i: " + str(i)
            # print "is reverse: " + str(self.is_reverse)
            # print "rho : " + str(rho)
            # print ""

            quaternion_world2goal = (
                self.path.poses[i].pose.orientation.x,
                self.path.poses[i].pose.orientation.y,
                self.path.poses[i].pose.orientation.z,
                self.path.poses[i].pose.orientation.w)
            euler_world2goal = tf.transformations.euler_from_quaternion(quaternion_world2goal, axes='sxyz')
            phi = euler_world2goal[2]

        ## Go to the goal
        if rho < 0.01:
            self.change_subgoal = True
            self.cmd_vel.linear.x = 0.0
            self.cmd_vel.angular.z = 0.0
        else:
            theta = robot.pose.angular.z
            alpha = self.wraptopi(atan2(dy, dx) - theta)
            beta = self.wraptopi(phi - alpha - theta)

            if self.is_reverse:
                self.cmd_vel.angular.z = k_alpha * self.wraptopi(alpha-pi) + k_beta * self.wraptopi(beta-pi)
                self.cmd_vel.linear.x = - k_rho * rho
            else:
                self.cmd_vel.angular.z = k_alpha * alpha + k_beta * beta
                self.cmd_vel.linear.x = k_rho * rho


        self.pub_cmd.publish(self.cmd_vel)

        self.u[0] = self.cmd_vel.linear.x
        self.u[1] = self.cmd_vel.angular.z

        if controller_debug:
            print "rho"
            print rho
            print "alpha"
            print alpha / pi * 180
            print "beta"
            print beta / pi * 180
            print "theta"
            print robot.pose.angular.z / pi * 180

        return self.u

    def goal_cb(self, path):
        self.path = path
        self.goal_i = 0
        self.idle = False
        self.change_subgoal = True
        self.seq_change = []
        self.seq_is_reverse = []

        if len(self.path.poses) > 3:
            # Detect change of direction
            for i in range(len(self.path.poses) - 2):
                byaw0 = atan2(self.path.poses[i+1].pose.position.y - self.path.poses[i].pose.position.y,
                              self.path.poses[i+1].pose.position.x - self.path.poses[i].pose.position.x)
                byaw1 = atan2(self.path.poses[i+2].pose.position.y - self.path.poses[i+1].pose.position.y,
                              self.path.poses[i+2].pose.position.x - self.path.poses[i+1].pose.position.x)
                if abs(self.wraptopi(byaw1 - byaw0)) > pi / 2:
                    self.seq_change.append(i+1)

            for i in range(len(self.path.poses) - 1):
                byaw0 = atan2(self.path.poses[i+1].pose.position.y - self.path.poses[i].pose.position.y,
                              self.path.poses[i+1].pose.position.x - self.path.poses[i].pose.position.x)

                quaternion_world2goal = (
                    self.path.poses[i].pose.orientation.x,
                    self.path.poses[i].pose.orientation.y,
                    self.path.poses[i].pose.orientation.z,
                    self.path.poses[i].pose.orientation.w)
                euler_world2goal = tf.transformations.euler_from_quaternion(quaternion_world2goal, axes='sxyz')
                phi = euler_world2goal[2]

                if abs(self.wraptopi(phi - byaw0)) > pi / 2:
                    self.seq_is_reverse.append(True)
                else:
                    self.seq_is_reverse.append(False)
            self.seq_is_reverse.append(self.seq_is_reverse[-1])

            for i in self.seq_change:
                self.seq_is_reverse[i] = self.seq_is_reverse[i-1]

            self.i_max = len(self.path.poses)-1
            self.seq_change.append(self.i_max)

    @staticmethod
    def wraptopi(angle):
        angle = (angle + pi) % (2 * pi) - pi
        return angle


class GlobalPathPlanner:
    def __init__(self):
        self.map = OccupancyGrid()
        self.pub_global_path = rospy.Publisher('/robot0/global_path', Path, queue_size=1)
        self.pub_path_graph = rospy.Publisher('/robot0/path_graph', Twist, queue_size=100)
        self.path = Path()
        self.path_point = Twist()


        # Robot object.
        self.robot = Robot()

    @staticmethod
    def wraptopi(angle):
        angle = (angle + pi) % (2 * pi) - pi
        return angle

    def goal_cb(self, goal):
        quaternion_world2goal = (
            goal.pose.orientation.x,
            goal.pose.orientation.y,
            goal.pose.orientation.z,
            goal.pose.orientation.w)
        euler_angle = tf.transformations.euler_from_quaternion(quaternion_world2goal, axes='sxyz')
        rospy.loginfo("Goal set to x = {0} [m], y = {1} [m], yaw = {2} [deg])"
                      .format(str(goal.pose.position.x), str(goal.pose.position.y), str(euler_angle[2]/pi*180)))

        # A star algorithm
        if is_a_star:
            sx = self.robot.pose.position.x  # [m]
            sy = self.robot.pose.position.y  # [m]
            gx = goal.pose.position.x  # [m]
            gy = goal.pose.position.y  # [m]

            grid_size = self.map.info.resolution   # [m]
            offset_x = self.map.info.origin.position.x
            offset_y = self.map.info.origin.position.y
            robot_size = 0.5  # [m]

            ox, oy = [], []

            for i in range(self.map.info.height):
                for j in range(self.map.info.width):
                    if self.map.data[i * self.map.info.width + j] > 0:
                        ox.append(j*grid_size + offset_x)
                        oy.append(i*grid_size + offset_y)

            if show_animation:
                plt.plot(ox, oy, ".k")
                plt.plot(sx, sy, "xr")
                plt.plot(gx, gy, "xb")
                plt.grid(True)
                plt.axis("equal")

            rx, ry = a_star.a_star_planning(sx, sy, gx, gy, ox, oy, grid_size, robot_size)
            ryaw = []
            for i in range(len(rx)-1):
                ryaw.append(atan2(ry[i]-ry[i+1], rx[i]-rx[i+1]))
            ryaw.append(ryaw[-1])

            if show_animation:
                plt.plot(rx, ry, "-r")
                plt.show()

        if is_reeds_sheep:
            start_x = self.robot.pose.linear.x  # [m]
            start_y = self.robot.pose.linear.y  # [m]
            start_yaw = self.robot.pose.angular.z  # [rad]

            end_x = goal.pose.position.x  # [m]
            end_y = goal.pose.position.y  # [m]
            end_yaw = euler_angle[2]  # [rad]

            curvature = 1.0
            step_size = 0.05

            rx, ry, ryaw, mode, clen = reeds_shepp.reeds_shepp_path_planning(
                start_x[0], start_y[0], start_yaw[0], end_x, end_y, end_yaw, curvature, step_size)

            # Detect backward direction
            bx = []
            by = []
            for i in range(len(rx) - 1):
                byaw = atan2(ry[i + 1] - ry[i], rx[i + 1] - rx[i])
                delta_angle = self.wraptopi(byaw - ryaw[i])
                if abs(delta_angle) > pi / 2:
                    bx.append(rx[i])
                    by.append(ry[i])

            if show_animation:
                plt.cla()
                plt.plot(rx, ry, label="final course " + str(mode))
                plt.plot(bx, by, "or")
                reeds_shepp.plot_arrow(start_x, start_y, start_yaw)
                reeds_shepp.plot_arrow(end_x, end_y, end_yaw)

                plt.legend()
                plt.grid(True)
                plt.xlim((min(start_x, end_x) - 3, max(start_x, end_x) + 3))
                plt.ylim((min(start_y, end_y) - 3, max(start_y, end_y) + 3))
                plt.show()

        for ix, iy, iyaw in zip(rx, ry, ryaw):
            quaternion_path = tf.transformations.quaternion_from_euler(0, 0, iyaw, axes='sxyz')
            pose = PoseStamped()
            pose.header.frame_id = "world"
            pose.pose.position.x = float(ix)
            pose.pose.position.y = float(iy)
            pose.pose.position.z = 0.0  # Debug: float(iyaw/pi*180)
            pose.pose.orientation.x = float(quaternion_path[0])
            pose.pose.orientation.y = float(quaternion_path[1])
            pose.pose.orientation.z = float(quaternion_path[2])
            pose.pose.orientation.w = float(quaternion_path[3])
            pose.header.seq = self.path.header.seq + 1
            self.path.header.frame_id = "world"
            self.path.header.stamp = rospy.Time.now()
            pose.header.stamp = self.path.header.stamp
            self.path.poses.append(pose)

            # Publish path for graphics
            self.path_point.linear.x = ix
            self.path_point.linear.y = iy
            self.path_point.linear.z = iyaw
            self.pub_path_graph.publish(self.path_point)

        self.pub_global_path.publish(self.path)

        self.path = Path()

    def tf_cb(self, event):
        # Send transform between /world and /map
        br = tf.TransformBroadcaster()
        br.sendTransform((0, 0, 0),
                         (0, 0, 0, 1),
                         rospy.Time.now(),
                         "/map",
                         "/world")


def main():
    # Init gpss_filter node.
    rospy.init_node("gpss_filter")

    # Robot object definition.
    robot = Robot()

    ## Create global path planner object
    gpp = GlobalPathPlanner()
    gpp.robot = robot

    # Read static map
    rospy.wait_for_service('static_map')

    try:
        map_server = rospy.ServiceProxy('static_map', GetMap)
        gpp.map = map_server().map
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e

    # Timer to refresh TF transform between world and map
    rospy.Timer(rospy.Duration(1.0 / 4), gpp.tf_cb)

    # Subscribe to goal topic published by RVIZ
    rospy.Subscriber("/move_base_simple/goal", PoseStamped, gpp.goal_cb)

    ## Add filter object
    sf = SensorsFilter()
    sf.robot = robot

    # Subscribe to Odometry and ALVAR tag topics.
    odom_sub = message_filters.Subscriber("/robot0/diff_drive_controller/odom", Odometry)
    marker_sub = message_filters.Subscriber("/ar_pose_marker", AlvarMarkers)

    # Synchronize topics with message_filter.
    ts = message_filters.ApproximateTimeSynchronizer([odom_sub, marker_sub], 1, 0.010, allow_headerless=False)
    ts.registerCallback(sf.fusion)

    #rospy.Subscriber("/move_base_simple/goal", PoseStamped, sf.robot.goal_cb)
    rospy.Subscriber("/robot0/global_path", Path, robot.controller.goal_cb)

    # Blocks until ROS node is shutdown.
    rospy.spin()


if __name__ == "__main__":
    main()