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
from PIL import Image, ImageDraw, ImageFont

def combine_images_with_text(image1_path, image2_path, text1, text2, output_path):
    """
    Combine two images side by side with custom text above each image and save the result.
    
    Parameters:
    - image1_path: Path to the first image (PNG)
    - image2_path: Path to the second image (PNG)
    - text1: Text to display above the first image
    - text2: Text to display above the second image
    - output_path: Path where the combined image will be saved
    """
    # Open the two PNG images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # Get the dimensions of the images
    width1, height1 = image1.size
    width2, height2 = image2.size

    # Resize image2 to have the same height as image1
    if height1 != height2:
        new_height = height1
        image2 = image2.resize((int(width2 * new_height / height2), new_height))

    # Get the new dimensions after resizing
    width2, height2 = image2.size
    
    # Define font and size (you can choose a different font file if necessary)
    font = ImageFont.truetype("dejavu-sans-ttf-2.37/ttf/DejaVuSans.ttf", size=40)

    # Calculate the size of the new image (width is the sum of the two image widths, height includes text and images)
    text_height = font.size  # Height of text for image
    new_width = width1 + width2
    new_height = max(height1, height2) + text_height
    # Create a new image with the combined size and a white background
    new_image = Image.new('RGB', (new_width, new_height), color=(255, 255, 255))  # White background

    # Create a drawing context for adding text
    draw = ImageDraw.Draw(new_image)

    # Draw the text above each image
    draw.text((width1 // 2 - text_height // 2, 0), text1, font=font, fill=(0, 0, 0))
    draw.text(((width1 + width2 // 2 - text_height // 2)-50, 0), text2, font=font, fill=(0, 0, 0))

    # Paste the first image below its text
    new_image.paste(image1, (0, text_height))

    # Paste the second image below its text
    new_image.paste(image2, (width1, text_height))

    # Save the new combined image
    new_image.save(output_path)

    print(f"Images combined with text and saved as '{output_path}'")


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
    # Get properties of regions
    props = regionprops(mask, intensity_image=image)

    # Create a 2D array to store the coloured regions
    height, width = image.shape
    coloured_regions = np.zeros((height, width))

    for region in props:
        region_label = region.label
        
        # Get the mask for the current region
        region_mask = (mask == region_label)
        
        # Assign the mean intensity of the region to the corresponding pixels
        coloured_regions[region_mask] = region.mean_intensity

    cmap = plt.cm.viridis 
    newcolors = cmap(np.linspace(0, 1, cmap.N))

    # Set the first color (representing 0) to be transparent
    newcolors[0, :] = (0, 0, 0, 0)  # Transparent color (RGBA)
    cmap_new = mcolors.ListedColormap(newcolors)

    fig, ax = plt.subplots(figsize=(10, 10))
    norm = plt.Normalize(vmin=np.min(image), vmax=np.max(image))
    ax.imshow(image, cmap='gray', interpolation='nearest', alpha=0.6, norm=norm)
    norm = plt.Normalize(vmin=np.min(coloured_regions), vmax=np.max(coloured_regions))
    im = ax.imshow(coloured_regions, interpolation='nearest', alpha=0.5,cmap=cmap_new, norm=norm)

    ax.axis('off')

    # Add colorbar for regions
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Mean Intensity')
    #plt.show()
    output_file = os.path.join(output_path, filename)
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close() 

    return coloured_regions


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

    ax.axis('off')

    # Add colorbar for the label mask
    #cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    #cbar.set_label('Label Intensity')  # Optional label for the colorbar

    output_file = os.path.join(output_path, 'cellpose.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()  # Close the plot to avoid memory leaks


def process_images(input_folder, output_folder, diameter=50):
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
            number_of_nuclei = len(np.unique(expanded_intensity_labels)) - 1
            
            percentile=np.percentile(expanded_intensity_labels[expanded_intensity_labels!=0],95)
            number_of_macrophages=len(np.unique(expanded_intensity_labels[expanded_intensity_labels>percentile]))-1
            with open (os.path.join(output_path,'nuclei_count.txt'),'w') as f:
                f.write(f"Intensity threshold (90th percentile): {percentile}\n")
                f.write(f"Number of nuclei: {number_of_nuclei}\n")
                f.write(f"Number of macrophages: {number_of_macrophages}\n")
            
            text1 = f'Nuclei: {number_of_nuclei}'
            text2= f'Macrophages: {number_of_macrophages}, Intensity threshold: {percentile:.2f}'
            combine_images_with_text(os.path.join(output_path, 'cellpose.png'), os.path.join(output_path, 'expanded_intensities.png'), text1, text2, os.path.join(output_folder, f'{filename}_results.png'))

    javabridge.kill_vm()


# # Lesion
# input_folder = '/home/laura/WMS_Files/ProjectSupport/NM_MacrophageSegmentation/Lesion m4.2 ft 200224 '
# output_folder = '/home/laura/Desktop/output/lesion'
# process_images(input_folder, output_folder, diameter=50)

# Uterus
input_folder = '/home/laura/WMS_Files/ProjectSupport/NM_MacrophageSegmentation/Uterus m3 w4 170424/Uterus m3 w4 170424'
output_folder = '/home/laura/Desktop/output/uterus'
process_images(input_folder, output_folder, diameter=75)