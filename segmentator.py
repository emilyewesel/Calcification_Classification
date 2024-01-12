#This is the code used to segment the images
#I do this outside of the prediction loop to save time
# There is no image number 50, so I had to do this in two batches. 

for i in range(51, 215):
    inputfile = f'M0_{i}.nii.gz'
    outputfile = f'segmentations/M0_{i}_heart.nii.gz'
    # ! TotalSegmentator -i $inputfile -o $outputfile --roi_subset heart

import numpy as np
import nibabel as nib

margin = 30  # Set the desired margin

for i in range(211, 215):
    input_file = f'M0_{i}.nii.gz'
    output_file = f'cropped/heart_cropped{i}.nii.gz'

    # Load the original image
    original_image = nib.load(input_file).get_fdata()

    # Load the segmentation mask (make sure it's binary, with the heart as foreground)
    segmentation_mask = nib.load(f'segmentations/M0_{i}_heart.nii.gz/heart.nii.gz').get_fdata()

    # Apply the segmentation mask to the original image
    cropped_image = original_image * segmentation_mask

    # Get the minimum and maximum values along each axis (x, y, z)
    non_zero_indices = np.argwhere(cropped_image != 0)
    print(non_zero_indices)
    min_coords = np.min(non_zero_indices, axis=0) - margin
    max_coords = np.max(non_zero_indices, axis=0) + margin

    # Ensure the coordinates are within the image bounds
    min_coords = np.clip(min_coords, 0, np.array(original_image.shape) - 1)
    max_coords = np.clip(max_coords, 0, np.array(original_image.shape) - 1)

    # Extract the region of interest from the original image
    cropped_3d_image = original_image[
        min_coords[0]:max_coords[0] + 1,
        min_coords[1]:max_coords[1] + 1,
        min_coords[2]:max_coords[2] + 1
    ]

    print(i)
    print(f"Bottom Left Front: {min_coords}")
    print(f"Top Right Back: {max_coords}")
    print(f"Cropped Image Shape: {cropped_3d_image.shape}")

    # Save the cropped 3D image
    nib.save(nib.Nifti1Image(cropped_3d_image, affine=None), output_file)
