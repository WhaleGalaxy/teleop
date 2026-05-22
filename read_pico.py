import time
import xrobotoolkit_sdk as xrt

xrt.init()
print("init OK")
time.sleep(1.0)

try:
    while True:
        ts = xrt.get_time_stamp_ns()

        rp = xrt.get_right_controller_pose()
        lp = xrt.get_left_controller_pose()
        hp = xrt.get_headset_pose()

        ra = xrt.get_right_axis()
        la = xrt.get_left_axis()

        print("\033[2J\033[H", end="")
        print("timestamp_ns :", ts)
        print("right_pose   :", rp)
        print("left_pose    :", lp)
        print("headset_pose :", hp)
        print("right_axis   :", ra)
        print("left_axis    :", la)
        print("right_trigger:", xrt.get_right_trigger())
        print("right_grip   :", xrt.get_right_grip())
        print("left_trigger :", xrt.get_left_trigger())
        print("left_grip    :", xrt.get_left_grip())
        print("A B X Y      :", xrt.get_A_button(), xrt.get_B_button(), xrt.get_X_button(), xrt.get_Y_button())
        print("right_click  :", xrt.get_right_axis_click())
        print("left_click   :", xrt.get_left_axis_click())

        time.sleep(0.05)

finally:
    xrt.close()
