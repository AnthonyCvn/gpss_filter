#!/usr/bin/env python
import roslib; roslib.load_manifest('gpss_ekf')
import rospy
import smach
import smach_ros

from std_msgs.msg import Empty
from visualization_msgs.msg import Marker

# global parameters
NUM_FRAMES = 2


def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func

    return decorate


@static_var("counter", NUM_FRAMES)
def calib_monitor_cb(userdata, msg):
    rospy.loginfo("pose: " + str(msg.pose.position.x))
    calib_monitor_cb.counter -= 1
    rospy.loginfo("Calibration frame counter: " + str(NUM_FRAMES - calib_monitor_cb.counter))
    if calib_monitor_cb.counter <= 0:
        calib_monitor_cb.counter = NUM_FRAMES
        return False
    else:
        return True


@static_var("counter", NUM_FRAMES)
def filter_monitor_cb(userdata, msg):
    rospy.loginfo("pose: " + str(msg.pose.position.x))
    filter_monitor_cb.counter -= 1
    rospy.loginfo("calibration counter: " + str(filter_monitor_cb.counter))
    if filter_monitor_cb.counter <= 0:
        filter_monitor_cb.counter = NUM_FRAMES
        return False
    else:
        return True


def main():
    rospy.init_node("gpss_ekf")

    sm = smach.StateMachine(outcomes=['PREEMPTED', 'DONE'])
    with sm:
        smach.StateMachine.add('CALIBRATION', smach_ros.MonitorState("/visualization_marker", Marker, calib_monitor_cb),
                               transitions={'invalid': 'FILTER', 'valid': 'CALIBRATION', 'preempted': 'CALIBRATION'})

        smach.StateMachine.add('FILTER', smach_ros.MonitorState("/visualization_marker", Marker, filter_monitor_cb),
                               transitions={'invalid': 'FILTER', 'valid': 'FILTER', 'preempted': 'FILTER'})

    sis = smach_ros.IntrospectionServer('smach_server', sm, '/SM_ROOT')
    sis.start()
    sm.execute()
    rospy.spin()
    sis.stop()


if __name__ == "__main__":
    main()
