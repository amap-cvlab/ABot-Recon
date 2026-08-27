import os
import struct
import numpy as np
import collections

# Data containers
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

# Map COLMAP model IDs to parameter counts
# 0=SIMPLE_PINHOLE, 1=PINHOLE, 2=SIMPLE_RADIAL, 3=RADIAL, 4=OPENCV, etc.
CAMERA_MODEL_PARAMS = {0:3, 1:4, 2:4, 3:5, 4:8, 5:9, 6:10, 7:12, 8:5, 9:4, 10:6}


def read_cameras_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_PARAMS[model_id]
            params = read_next_bytes(fid, 8*num_params, "d"*num_params)
            cameras[camera_id] = Camera(id=camera_id, model=model_id,
                                      width=width, height=height,
                                      params=np.array(params))
    return cameras

def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
                
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            x_y_id_s = read_next_bytes(fid, 24*num_points2D, "ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            
            images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec,
                                     camera_id=camera_id, name=image_name,
                                     xys=xys, point3D_ids=point3D_ids)
    return images

def read_points3D_binary(path_to_model_file):
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, 43, "QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = binary_point_line_properties[7]
            track_length = read_next_bytes(fid, 8, "Q")[0]
            track_elems = read_next_bytes(fid, 8*track_length, "ii"*track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(id=point3D_id, xyz=xyz, rgb=rgb,
                                           error=error, image_ids=image_ids,
                                           point2D_idxs=point2D_idxs)
    return points3D

# Helper to read bytes
def read_next_bytes(fid, num_bytes, format_char_sequence):
    data = fid.read(num_bytes)
    return struct.unpack("<" + format_char_sequence, data)

def get_c2w_matrix(image):
    # 1. Extract World-to-Camera Rotation (R_w2c) from Quaternion
    # COLMAP Quaternion is [w, x, y, z]
    w, x, y, z = image.qvec
    R_w2c = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    
    # 2. Extract World-to-Camera Translation (t_w2c)
    t_w2c = image.tvec

    # 3. Compute Camera-to-World (Inverse)
    # Formula: T_c2w = [ R_w2c^T   -R_w2c^T * t_w2c ]
    #                  [    0             1        ]
    R_c2w = R_w2c.T
    t_c2w = -R_c2w @ t_w2c

    # 4. Pack into 4x4 Matrix
    c2w = np.eye(4)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = t_c2w
    
    return c2w

def read_poses_from_colmap(colmap_model_path):
    # list of poses sorted by image id
    # return:  (N, 4, 4)
    images = read_images_binary(os.path.join(colmap_model_path, "images.bin"))
    img_and_poses = []
    for img_id, img_data in images.items():
        pose = get_c2w_matrix(img_data)
        img_and_poses.append((img_data.name, pose))

    poses = sorted(img_and_poses, key=lambda x: x[0])
    poses = np.stack([np.array(p[1]) for p in poses], axis=0)  # (N, 4, 4)
    return poses
