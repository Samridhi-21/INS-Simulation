import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

# load real IMU dataset
data = pd.read_csv("imu_data.csv")

time = data["time"]
ax = data["ax"]
ay = data["ay"]
az = data["az"]
gx = data["gx"]
gy = data["gy"]
gz = data["gz"]
# add slight noise
ax = ax + np.random.normal(0, 0.02, len(ax))
ay = ay + np.random.normal(0, 0.02, len(ay))
az = az + np.random.normal(0, 0.02, len(az))

# smoothing
ax = np.convolve(ax, np.ones(3)/3, mode='same')
ay = np.convolve(ay, np.ones(3)/3, mode='same')
az = np.convolve(az, np.ones(3)/3, mode='same')



plt.plot(time, ax)
plt.plot(time, ay)
plt.plot(time, az)

# gyroscope data simulation

gx = np.sin(time)*0.5 + np.random.normal(0,0.1,len(time))
gy = np.cos(time)*0.5 + np.random.normal(0,0.1,len(time))
gz = np.sin(2*time)*0.5 + np.random.normal(0,0.1,len(time))
# smoothing gyroscope data
gx = np.convolve(gx, np.ones(5)/5, mode='same')
gy = np.convolve(gy, np.ones(5)/5, mode='same')
gz = np.convolve(gz, np.ones(5)/5, mode='same')

plt.show()

# velocity calculation (integration of acceleration)
vx = np.cumsum(ax)
vy = np.cumsum(ay)
vz = np.cumsum(az)

plt.figure()
plt.plot(time, vx)
plt.plot(time, vy)
plt.plot(time, vz)

plt.title("Velocity from Accelerometer")
plt.xlabel("Time")
plt.ylabel("Velocity")

plt.show()

# position calculation
px = np.cumsum(vx)
py = np.cumsum(vy)
pz = np.cumsum(vz)

plt.figure()
plt.plot(time, px)
plt.plot(time, py)
plt.plot(time, pz)

plt.title("Position from Velocity")
plt.xlabel("Time")
plt.ylabel("Position")

plt.show()

plt.figure()
plt.plot(px, py)

plt.title("Object Trajectory")
plt.xlabel("X Position")
plt.ylabel("Y Position")

plt.show()

plt.figure()
plt.plot(time, gx)
plt.plot(time, gy)
plt.plot(time, gz)

plt.title("Gyroscope Data")
plt.xlabel("Time")
plt.ylabel("Angular Velocity")

plt.show()

# orientation calculation from gyroscope
theta_x = np.cumsum(gx)
theta_y = np.cumsum(gy)
theta_z = np.cumsum(gz)

plt.figure()
plt.plot(time, theta_x)
plt.plot(time, theta_y)
plt.plot(time, theta_z)

plt.title("Orientation from Gyroscope")
plt.xlabel("Time")
plt.ylabel("Angle")

plt.show()
# simple sensor fusion
fused_x = theta_x + px * 0.01
fused_y = theta_y + py * 0.01
fused_z = theta_z + pz * 0.01

plt.figure()
plt.plot(time, fused_x)
plt.plot(time, fused_y)
plt.plot(time, fused_z)

plt.title("Fused Orientation (Accelerometer + Gyroscope)")
plt.xlabel("Time")
plt.ylabel("Angle")

plt.show()
