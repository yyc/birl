import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors
import DrawTrajectory
import pic_to_gif

def draw_trajectories(demos):
    all_thresholds = np.arange(len(demos))
    envs = np.load("valid_environments.npy")[0]
    envs[0,0] = 0
    for i in range(len(demos)):
        temp_envs = np.array(envs)
        temp_demos = demos[0:i+1]
        latest_demo = demos[i]
        goal = latest_demo[len(latest_demo)-1,0]
        start = latest_demo[0,0]
        temp_envs[goal,0] = 1
        plt.figure(i, figsize=(24, 13.5), dpi=80)
        p1 = plt.subplot(1, 1, 1)
        colors = []
        for c in range(0,i):
            colors.append('darkgrey')
        colors.append('yellow')
        #['blue','green','red','cyan','magenta','yellow','black','white','blue']
        DrawTrajectory.draw_traj(p1, env=temp_envs, trajectories=temp_demos, grid_size=5,
                  arrow_colors=colors,
                  threshold_plt=False,
                  temp_threshold=all_thresholds, width=0.015, legend_plot=True,start_state = start)
        plt.savefig("demo_plots/demos_"+str(i)+".png")
    plt.close("all")
    pic_to_gif.convert_to_gif(len(demos))

start = 2
end = 5
all_bm = []
for i in range(start,end+1):
    all_bm.append(np.load("BestMetric"+str(i)+".npy"))
all_bm = np.array(all_bm)
plt.figure(44, figsize=(24, 13.5), dpi=80)
plt.plot(all_bm)
plt.savefig("demo_plots/metrics.png")

d = np.load("AllD5.npy")
draw_trajectories(d)



