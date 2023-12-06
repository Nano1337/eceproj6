import ujson as json
import numpy as np

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from LaplacianInterpolation import laplacianInterpolation
from DisplayVolume import *

# Load Project6.json
with open('Project6.json', 'r') as file:
    project_data = json.load(file)

mr = project_data['mr']
mrsps = project_data['mrsps']


# Perform Laplacian Interpolation
# output is a [r, c, d] nested list
phi = laplacianInterpolation(mrsps['dim'][0], mrsps['dim'][1], mrsps['dim'][2], mrsps['nds'], mrsps['data'])
phi = np.array(phi)

# Ensure dimensions match
original_image = np.array(mr['data'])
assert phi.shape == original_image.shape, "Shape mismatch between the interpolated and original images."

# Calculate mean absolute difference
mean_abs_difference = np.mean(np.abs(phi - original_image))
print(f"Mean Absolute Difference: {mean_abs_difference}")

# Display the image with display volume
d = DisplayVolume()
# Set the image with default voxel size
voxsz = project_data['mr']['voxsz']  # Adjust as per your actual voxel size
d.SetImage(phi, voxsz)

# Display the image
d.Display()

