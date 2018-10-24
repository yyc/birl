import os
import os.path
import imageio
import sys
import datetime

end_file = 9

def get_files(target_dir):
    item_list = os.listdir(target_dir)

    file_list = list()
    for item in item_list:
        item_dir = os.path.join(target_dir,item)
        if os.path.isdir(item_dir):
            file_list += get_files(item_dir)
        else:
            if "true_r" in item_dir:
                file_list.append(item_dir)
    return file_list

def create_gif(filenames, duration,plot_type = "reward"):
    images = []
    for filename in filenames:
        print filename
        images.append(imageio.imread(filename))
    output_file = plot_type + 'Gif-%s.gif' % datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S')
    imageio.mimsave(output_file, images, duration=duration)

def convert_to_gif(end_file):
    filenames=[]
    for i in range(0,end_file):
        filenames.append("demo_plots/demos_"+str(i)+".png")
    create_gif(filenames,3,"reward")

