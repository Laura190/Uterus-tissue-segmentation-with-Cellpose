import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from skimage.segmentation import expand_labels
from skimage.color import label2rgb
from skimage.io import imread
from scipy.spatial import cKDTree, Voronoi
from cellpose import models
import javabridge
import bioformats
import matplotlib.colors as mcolors
from scipy.spatial import Voronoi, voronoi_plot_2d

def compute_label_intensities(mask, image, output_path, filename, expand=10):
    """
    Compute intensities for each labeled region in the mask based on the image.

    Parameters:
    - mask: 2D numpy array of labeled regions (each region has a unique label)
    - image: 2D numpy array of the image (intensities)
    - output_path: Path to save the output plot

    Returns:
    - labeled_regions: A 2D array where each pixel is assigned to its mean intensity
    """
    mask = expand_labels(mask, expand)
    # Get properties of labeled regions (centroids, area, etc.)
    props = regionprops(mask, intensity_image=image)

    # Create a 2D array to store the labeled regions
    height, width = image.shape
    coloured_regions = np.zeros((height, width))

    # Calculate the region intensities for each label using regionprops' mean_intensity
    for region in props:
        region_label = region.label
        
        # Get the mask for the current region
        region_mask = (mask == region_label)
        
        # Assign the region label to the corresponding pixels
        coloured_regions[region_mask] = region.mean_intensity

    cmap = plt.cm.viridis  # You can choose any colormap here, e.g., viridis, plasma, etc.
    newcolors = cmap(np.linspace(0, 1, cmap.N))

    # Set the first color (representing 0) to be transparent (or black, as you wish)
    newcolors[0, :] = (0, 0, 0, 0)  # Transparent color (RGBA)
    cmap_new = mcolors.ListedColormap(newcolors)

    fig, ax = plt.subplots(figsize=(10, 10))
    norm = plt.Normalize(vmin=np.min(image), vmax=np.max(image))
    ax.imshow(image, cmap='gray', interpolation='nearest', alpha=0.6, norm=norm)
    norm = plt.Normalize(vmin=np.min(coloured_regions), vmax=np.max(coloured_regions))
    im = ax.imshow(coloured_regions, interpolation='nearest', alpha=0.5,cmap=cmap_new, norm=norm)

    # Add colorbar for the label mask
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Label Intensity')  # Optional label for the colorbar
    #plt.show()
    output_file = os.path.join(output_path, filename)
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()  # Close the plot to avoid memory leaks

    return coloured_regions


def compute_voronoi_from_mask(mask, image):
    """
    Compute Voronoi tessellation from the mask and calculate the intensity of each Voronoi region based on the image.
    """
    props = regionprops(mask)
    centroids = np.array([prop.centroid for prop in props])[:, [1, 0]]  # Convert to (x, y)
    tree = cKDTree(centroids)

    voronoi_regions = np.zeros(image.shape, dtype=int)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            _, region = tree.query([x, y])
            voronoi_regions[y, x] = region

    return voronoi_regions


def plot_voronoi_with_intensities(image, voronoi_regions, region_intensities, output_path):
    """
    Plot Voronoi regions on the image, colored by the average intensity of each region.
    """
    colormap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(region_intensities), vmax=max(region_intensities))

    colored_regions = np.zeros((image.shape[0], image.shape[1], 3))  # RGB image
    for region, intensity in enumerate(region_intensities, start=1):
        region_mask = (voronoi_regions == region)
        color = colormap(norm(intensity))[:3]
        colored_regions[region_mask] = color

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap='gray', interpolation='nearest', alpha=0.6)
    im = ax.imshow(colored_regions, interpolation='nearest', alpha=0.5)

    # Add colorbar for the label mask
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Label Intensity')  # Optional label for the colorbar

    #plt.show()
    output_file = os.path.join(output_path, 'voronoi_intensities.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()  # Close the plot to avoid memory leaks


def max_project_z(image_stack):
    """
    Max projection of a 3D image stack along the z-axis.
    """
    return np.max(image_stack, axis=0)


def plot_image_with_mask(image, mask, colormap, alpha, maxint, minint, output_path):
    """
    Plot an image with a label mask overlay.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap='gray', interpolation='nearest', vmin=minint, vmax=(maxint-minint)//2+minint)
    
    labels = label2rgb(mask, bg_label=0, bg_color=(0, 0, 0))
    im= ax.imshow(labels, cmap=colormap, interpolation='nearest', alpha=alpha)

    # Add colorbar for the label mask
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Label Intensity')  # Optional label for the colorbar

    output_file = os.path.join(output_path, 'cellpose.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()  # Close the plot to avoid memory leaks


def process_images(input_folder, output_folder):
    """
    Process images in the input folder, apply Cellpose, and save results to the output folder.
    """
    javabridge.start_vm(class_path=bioformats.JARS)

    model = models.Cellpose(model_type='cyto3')
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.oir'):
        #if filename.endswith('.tif'):
            filepath = os.path.join(input_folder, filename)

            # Use Bio-Formats to read the image
            with bioformats.ImageReader(filepath) as reader:
                z_size, c_size, t_size = reader.rdr.getSizeZ(), reader.rdr.getSizeC(), reader.rdr.getSizeT()
                x_size, y_size = reader.rdr.getSizeX(), reader.rdr.getSizeY()

                image_data = np.zeros((z_size, c_size, t_size, x_size, y_size), dtype=np.uint16)
                for t in range(t_size):
                    for z in range(z_size):
                        for c in range(c_size):
                            image_data[z, c, t] = reader.read(c=c, z=z, t=t, series=None, rescale=False)
            #image_data = imread(os.path.join(input_folder, filename))

            # Max projection of channels
            channel_1 = np.squeeze(max_project_z(image_data[:, 0, :, :]))  # Channel 1 (Green)
            channel_3 = np.squeeze(max_project_z(image_data[:, 1, :, :]))  # Channel 3 (Magenta)

            # Run Cellpose on the max projected image
            masks, _, _, _ = model.eval(channel_1, diameter=50, channels=[2, 1])
            
            output_path = os.path.join(output_folder, filename)
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            plot_image_with_mask(channel_1, masks, colormap='prism', alpha=0.4, maxint=np.max(channel_1), minint=np.min(channel_1),output_path=output_path)

            # Compute Voronoi tessellation from the mask and plot it
            #voronoi_regions = compute_voronoi_from_mask(masks, channel_3)
            #voronoi_intensity_labels=compute_label_intensities(voronoi_regions, channel_3, output_path, 'voronoi_intensities.png',expand=0)

            # Compute region intensities from the mask and plot the labeled regions
            expanded_intensity_labels=compute_label_intensities(masks, channel_3, output_path, 'expanded_intensities.png', expand=10)
            number_of_nuclei=np.unique(expanded_intensity_labels)-1
            mean_nonzero=np.mean(expanded_intensity_labels[expanded_intensity_labels>0])
            number_of_macrophages=np.unique(expanded_intensity_labels[expanded_intensity_labels>mean_nonzero])-1
            print('Number of nuclei:',number_of_nuclei)
            print('Number of macrophages:',number_of_macrophages)

    javabridge.kill_vm()


# Example usage:
input_folder = '/home/laura/WMS_Files/ProjectSupport/NM_MacrophageSegmentation/Lesion m4.2 ft 200224 '
output_folder = '/home/laura/Desktop/output/lesion'
process_images(input_folder, output_folder)
