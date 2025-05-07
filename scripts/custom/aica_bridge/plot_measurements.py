import matplotlib.pyplot as plt
import numpy as np

def read_force_data_and_plot(filename: str = "force_data.txt", folder="source/isaaclab_assets/data/measurements/"):
    data = np.loadtxt(f"{folder}{filename}", delimiter=",")
    time = np.arange(data.shape[0]) * 0.005  # Assuming a time step of 0.005 seconds

    plt.figure(figsize=(10, 6))
    plt.plot(time, data[:, 0], label="Force X")
    plt.plot(time, data[:, 1], label="Force Y")
    plt.plot(time, data[:, 2], label="Force Z")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.title("Force Data Over Time")
    plt.legend()
    plt.grid()
    plt.savefig(f"{folder}force_data_plot.png")



read_force_data_and_plot("force_data.txt", "source/isaaclab_assets/data/measurements/")