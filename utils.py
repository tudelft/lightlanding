def scale_camera_matrix(K, s):
    K_new = K.copy().astype(float)
    K_new[0, 0] *= s
    K_new[1, 1] *= s
    K_new[0, 2] *= s
    K_new[1, 2] *= s
    return K_new

def rotate_intrinsics_180(K, image_width, image_height):
    """
    Update camera intrinsics after rotating image by 180 degrees.

    Parameters
    ----------
    K : (3,3) ndarray
        Original intrinsic matrix.
    image_width : int
    image_height : int

    Returns
    -------
    K_rot : (3,3) ndarray
        Updated intrinsic matrix.
    """

    K_rot = K.copy().astype(np.float64)

    K_rot[0, 2] = image_width  - 1 - K[0, 2]   # cx
    K_rot[1, 2] = image_height - 1 - K[1, 2]   # cy

    return K_rot
