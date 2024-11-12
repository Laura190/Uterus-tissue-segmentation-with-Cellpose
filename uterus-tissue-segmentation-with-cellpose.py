import os
import numpy as np
from skimage import io
from cellpose import models, io as cp_io
from oiffile import imread
import javabridge
import bioformats
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def max_project_z(image_stack):
    return np.max(image_stack, axis=0)

def create_black_to_pink_cmap():
    # Create a custom colormap from black to purple/pink
    colors = [(0, 0, 0), (0.5, 0, 0.5), (1, 0, 1)]  # Black -> Purple -> Pink
    n_bins = 100  # Number of bins for interpolation
    cmap_name = 'black_to_pink'
    return LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

def create_black_to_green_cmap():
    # Create a custom colormap from black to green
    colors = [(0, 0, 0), (0, 1, 0)]  # Black -> Purple -> Pink
    n_bins = 100  # Number of bins for interpolation
    cmap_name = 'black_to_green'
    return LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

def process_images(input_folder, output_folder):
    
    # Start the Java VM (this step requires Java to be installed)
    javabridge.start_vm(class_path=bioformats.JARS)

    model = models.Cellpose(model_type='cyto3')
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    print(os.listdir(input_folder))
    for filename in os.listdir(input_folder):
        if filename.endswith('.oir'):
            filepath = os.path.join(input_folder, filename)

            # Use Bio-Formats to read the image
            with bioformats.ImageReader(filepath) as reader:
                # Get the image metadata
                z_size = reader.rdr.getSizeZ()
                c_size = reader.rdr.getSizeC()
                t_size = reader.rdr.getSizeT()
                x_size = reader.rdr.getSizeX()
                y_size = reader.rdr.getSizeY()
                
                image_data = reader.read()
                
                # Read image stack
                image_data = np.zeros((z_size, c_size, t_size, x_size, y_size), dtype=np.uint16)
                for t in range(t_size):
                    for z in range(z_size):
                        for c in range(c_size):
                            image_data[z, c, t] = reader.read(c=c, z=z, t=t, series=None, rescale=False)

            print(filepath)
            print(z_size, c_size, y_size, x_size)    
            # Select channels 1 and 3 for max projection
            channel_1 = np.squeeze(max_project_z(image_data[:, 0, :, :]))  # Channel 1 (Green)
            channel_3 = np.squeeze(max_project_z(image_data[:, 2, :, :]))  # Channel 3 (Magenta)
    
            # Stack the selected channels along the last axis
            max_proj_image = np.stack((channel_1, channel_3), axis=-1)
            
            # Run Cellpose on the max projected image
            print('start cellpose')
            masks, flows, styles, diams = model.eval(max_proj_image, diameter=80, channels=[1, 2])
            print('end cellpose')

            outlines = models.utils.outlines_list(masks)
            
            # Set the background to dark using matplotlib style
            plt.style.use('dark_background')
            plt.figure(figsize=(8, 8))
            
            # Overlay channels 1 (green) and 3 (magenta) on a dark background
            # Green for channel 1
            black_to_green_cmap = create_black_to_green_cmap()
            plt.imshow(channel_1, cmap=black_to_green_cmap, alpha=0.7)  # Green channel with transparency
            # Custom black-to-pink for channel 3
            black_to_pink_cmap = create_black_to_pink_cmap()
            plt.imshow(channel_3, cmap=black_to_pink_cmap, alpha=0.7)  # Magenta channel with transparency
            
            # Plot outlines
            for outline in outlines:
                plt.plot(outline[:, 1], outline[:, 0], color='red', linewidth=0.5)
                
            plt.axis('off')  # Turn off axis labels

            # Save the image to output folder
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_outlines.png")
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.show()

    # Stop the Java VM when you're done
    javabridge.kill_vm()

input_folder = '/home/laura/WMS_Files/ProjectSupport/NM_MacrophageSegmentation/Uterus m3 w4 170424/Uterus m3 w4 170424/'
output_folder = '/home/laura/Desktop/output'
process_images(input_folder, output_folder)
