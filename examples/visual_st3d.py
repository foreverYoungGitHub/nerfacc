
import pathlib
import numpy as np
import imageio
from datasets.nerf_st3d import SubjectLoader

data_root_fp = str(pathlib.Path.home() / "data/st3d")
scene = "03122_554516"

test_dataset = SubjectLoader(
    subject_id=scene,
    root_fp=data_root_fp,
    split="eval",
    num_rays=None,
    device="cpu",
    # **test_dataset_kwargs,
)
coord = test_dataset.coord
import seaborn as sns
import matplotlib.pyplot as plt

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


H, W = coord.shape[:2]
# coord = coord.reshape(-1, 3)

z_vec = np.array((0,0,1))
y_vec = np.array((0,1,0))
x_vec = np.array((1,0,0))

def calculate_theta(vec, phi):
    return -np.arctan(vec[1]/(vec[2]*np.cos(phi) + vec[0]*np.sin(phi))) + np.pi
    
# x, y = 100, 500
x, y = 500, 1000
_theta = (1 - 2 * (x) / H) * np.pi / 2  # latitude
_phi = 2 * np.pi * (0.5 - (y) / W)  # longtitude
print(_theta)
print(_phi)
# print()
# print(angle_between(coord[0], x_vec))
vec = unit_vector(coord[x, y].astype(np.double))

print(vec)
_theta = np.arcsin(vec[1])
_phi = np.arccos(vec[0]/np.cos(_theta))
if vec[2] > 0:
    _phi = - _phi
print(_theta)
print(_phi)
x = np.round((- (_theta * 2 / np.pi) + 1) * H / 2) % H
y = np.round((0.5 - _phi / (2 * np.pi)) * W) % W
print(x)
print(y)

assert False

# sns.jointplot(x=coord[:,0], y=coord[:,1], kind='hex')
# plt.savefig("hex_01.png")

# plt.clf()
# sns.jointplot(x=coord[:,1], y=coord[:,2], kind='hex')
# plt.savefig("hex_12.png")

# plt.clf()
# sns.jointplot(x=coord[:,0], y=coord[:,2], kind='hex')
# plt.savefig("hex_02.png")

plt.clf()
sns.jointplot(x=coord[:,0], y=coord[:,2], kind='hist')
plt.savefig("hist_02.png")

# plt.clf()
# sns.jointplot(x=coord[:,1], y=coord[:,2], kind='hist')
# plt.savefig("hist_12.png")
# assert False

rgb = test_dataset.rays_rgb
rgb = rgb.reshape(coord.shape)

size = 512

# print(coord.max(), coord.min())
coord = (coord * (size // 2) + (size // 2) ).astype(np.int)
# print(coord.max(), coord.min())


# top to center
img = np.ones((size, size, 3))
for i in range(0, rgb.shape[0]//2):
    for j in range(0, rgb.shape[1]):
        x,y,z = coord[i,j]
        img[x,z] = rgb[i,j]
img = (img * 255).astype(np.uint8)
imageio.imwrite("test_top.png",img)

# bottom to center
img = np.ones((size, size, 3))
for i in range(rgb.shape[0]//2, rgb.shape[0]):
    for j in range(0, rgb.shape[1]):
        x,y,z = coord[i,j]
        img[x,z] = rgb[i,j]
img = (img * 255).astype(np.uint8)
imageio.imwrite("test_bottom.png",img)

# # left to center
# img = np.ones((size, size, 3))
# for i in range(0, rgb.shape[0]):
#     for j in range(0, rgb.shape[1]//2):
#         x,y,z = coord[i,j]
#         img[size-y,size-x] = rgb[i,j]
# img = (img * 255).astype(np.uint8)
# imageio.imwrite("test_left.png",img)

# # right to center
# img = np.ones((size, size, 3))
# for i in range(0, rgb.shape[0]):
#     for j in range(rgb.shape[1]//2, rgb.shape[1]):
#         x,y,z = coord[i,j]
#         img[size-y,size-x] = rgb[i,j]
# img = (img * 255).astype(np.uint8)
# imageio.imwrite("test_right.png",img)

