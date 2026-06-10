import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
import itertools
import math

def get_endpoints_of_a_noisy_line(points):
    # PCA direction
    center = points.mean(axis=0)
    X = points - center

    _, _, vt = np.linalg.svd(X, full_matrices=False)
    direction = vt[0]

    # Position of each point along the arm
    proj = X @ direction

    # End circles
    idx1 = np.argmin(proj)
    idx2 = np.argmax(proj)

    end1 = points[idx1]
    end2 = points[idx2]

    return end1, end2

def pose_from_colored_leds(fitered_circles, filteredcircles_avgcolor_sorted, new_K, dist_coeffs):
    green_arm = fitered_circles[filteredcircles_avgcolor_sorted[:4]] #first 4 LEDs based on min avg intensity
    amber_arm = fitered_circles[filteredcircles_avgcolor_sorted[4:]]  #other 4 LEDs based on min avg intensity
    
    green_circles_indices = green_arm[:,:2]
    amber_circles_indices = amber_arm[:,:2]

    # green_circles_indices_lexsorted = green_circles_indices[np.lexsort((green_circles_indices[:,1], green_circles_indices[:,0]))]
    # green_corners = green_circles_indices_lexsorted[0], green_circles_indices_lexsorted[-1]

    # amber_circles_indices_lexsorted = amber_circles_indices[np.lexsort((amber_circles_indices[:,1], amber_circles_indices[:,0]))]
    # amber_corners = amber_circles_indices_lexsorted[0], amber_circles_indices_lexsorted[-1]

    green_corners = get_endpoints_of_a_noisy_line(green_circles_indices)
    amber_corners = get_endpoints_of_a_noisy_line(amber_circles_indices)
                                                  
#    image_points_perms = np.array(list(itertools.permutations(np.vstack((amber_edges, green_edges)))))
    image_points_perms = np.array([   #all 4 possible combinations of corner correspondences since we don't know apriori which is which
    [amber_corners[0], amber_corners[1], green_corners[0], green_corners[1]],
    [amber_corners[1], amber_corners[0], green_corners[0], green_corners[1]],
    [amber_corners[0], amber_corners[1], green_corners[1], green_corners[0]],
    [amber_corners[1], amber_corners[0], green_corners[1], green_corners[0]]
    ], dtype=np.float32) 
    
    # 3D object points in meters
    object_points = np.array([
        [0.0,    0.0,    -0.230],  # corner, long amber+green arm, amber led
        [0.375,   0.0,  -0.230],  # short arm, last amber led
        [0.0,  -0.255,    0.0],  # long amber+green arm, first green led
        [0.0,  -0.630,    0.0],  # long amber+green arm, last green led
    ], dtype=np.float32)

    min_reproj_error = float('inf')
    image_points_best_config = None
    rvec_best = None
    tvec_best = None
    projected_points_best = None
    positive_depth_best = None

    for image_points in image_points_perms:
        success, positive_depth, reproj_err, rvec, tvec, projected_points = estimate_pose_nonplanar(object_points, image_points, new_K, dist_coeffs)
        if success and positive_depth and reproj_err < min_reproj_error:
            image_points_best_config = image_points
            min_reproj_error = reproj_err
            rvec_best = rvec
            tvec_best = tvec
            projected_points_best = projected_points
            positive_depth_best = positive_depth

    if image_points_best_config is not None:
        R_wld_to_cam, _ = cv2.Rodrigues(rvec_best)
        #print('tvec', tvec)
        T_wld_to_cam = np.eye(4)
        T_wld_to_cam[:3, :3] = R_wld_to_cam
        T_wld_to_cam[:3, 3] = tvec_best.flatten()

        # Invert transform
        T_cam_to_wld = np.linalg.inv(T_wld_to_cam)

        T_cam_to_drone = np.array([
            [ 0, -1,  0, 0],
            [ 1,  0,  0, 0],
            [ 0,  0,  1, 0],
            [ 0,  0,  0, 1],
        ])

        #T_drone_to_wld = T_cam_to_wld 
        T_drone_to_wld = T_cam_to_wld @ np.linalg.inv(T_cam_to_drone)

        cam_pos = T_drone_to_wld[:3, 3]
        cam_orient_quat = R.from_matrix(T_drone_to_wld[:3, :3]).as_quat()  # (x, y, z, w)
        pose_dict = {
            "success": True,
            "rvec": rvec_best,
            "tvec": tvec_best,
            "R": R_wld_to_cam,
            "camera_position": cam_pos,
            "camera_orientation": cam_orient_quat,
            "reprojection_error": min_reproj_error,
            "projected_points": projected_points_best,
            "positive_depth": positive_depth_best,
        }

        return image_points_best_config, object_points, pose_dict
    
    else:
        return None, None, {"success": False}

def order_l_shape_markers(circles):
    pts = np.array(circles)[:, :2].astype(np.float32)
    dist_matrix = cdist(pts, pts)
    
    # --- 1. Find the Corner using Local Geometry ---
    # For each point, find its two closest neighbors and calculate the angle
    # The corner will have neighbors forming roughly a 90-degree angle.
    best_corner_idx = -1
    min_angle_diff = float('inf')
    
    for i in range(len(pts)):
        # Get indices of two closest points (excluding self)
        nearest_indices = np.argsort(dist_matrix[i])[1:3]
        p1, p2 = pts[nearest_indices[0]], pts[nearest_indices[1]]
        
        # Vectors from current point to neighbors
        v1 = p1 - pts[i]
        v2 = p2 - pts[i]
        
        # Calculate angle between vectors
        unit_v1 = v1 / np.linalg.norm(v1)
        unit_v2 = v2 / np.linalg.norm(v2)
        dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
        angle = np.arccos(dot_product)
        
        # We want the angle closest to pi/2 (90 degrees)
        diff = abs(angle - np.pi/2)
        if diff < min_angle_diff:
            min_angle_diff = diff
            best_corner_idx = i

    corner = pts[best_corner_idx]
    # --- 2. Separate Arms ---
    other_indices = [i for i in range(8) if i != best_corner_idx]
    others = pts[other_indices]
    vectors = others - corner
    
    # Use the point furthest from the corner to define the "Long Arm" vector
    farthest_idx = np.argmax(np.linalg.norm(vectors, axis=1))
    long_vec_ref = vectors[farthest_idx]
    
    # Group points by checking alignment with the long_vec_ref
    # Points on the same arm will have a very high cosine similarity (near 1.0)
    cos_sims = np.dot(vectors, long_vec_ref) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(long_vec_ref))
    
    # The 4 points with the highest similarity belong to the long arm
    long_arm_mask = np.argsort(cos_sims)[-4:]
    short_arm_mask = np.argsort(cos_sims)[:3]
    
    long_indices = np.array(other_indices)[long_arm_mask]
    short_indices = np.array(other_indices)[short_arm_mask]
    
    # --- 3. Sort by distance from corner ---
    def sort_by_dist(idx_list):
        dists = np.linalg.norm(pts[idx_list] - corner, axis=1)
        return np.array(idx_list)[np.argsort(dists)]

    sorted_long = sort_by_dist(long_indices)
    sorted_short = sort_by_dist(short_indices)
    
    # Combine into final array [0=corner, 1-4=long, 5-7=short]
    final_indices = [best_corner_idx] + list(sorted_long) + list(sorted_short)
    
    image_points = pts[final_indices].astype(np.float32)
        
    # 3D object points in cm
    object_points = np.array([
        [0.0,  0.0,  0.230],   # corner
        [0.130, 0.0,  0.0],
        [0.255, 0.0,  0.0],
        [0.380, 0.0,  0.0],
        [0.505, 0.0,  0.0],   # long arm

        [0.0,  0.125, 0.230],
        [0.0,  0.250, 0.230],
        [0.0,  0.375, 0.230],   # short arm
    ], dtype=np.float32)

    info = {
    }

    return image_points, object_points, info    

def order_l_shape_markers_old(circles):
    """
    Orders 8 circles [x, y, r] into the L-shape convention.
    0: Corner
    1-4: Long arm (+X direction)
    5-7: Short arm (+Y direction)
    """
    # Extract only (x, y) coordinates
    pts = np.array(circles)[:, :2].astype(np.float32)
    
    # Calculate distance matrix between all pairs of points
    dist_matrix = cdist(pts, pts)
    
    # --- 1. Robust Corner Detection via 90-Degree Angle Analysis ---
    best_corner_idx = -1
    min_angle_diff = float('inf')
    
    for i in range(8):
        # Find the 2 nearest neighbors to point i (index 0 is the point itself)
        nearest_indices = np.argsort(dist_matrix[i])[1:3]
        p1, p2 = pts[nearest_indices[0]], pts[nearest_indices[1]]
        
        # Build vectors from point i to these two neighbors
        v1 = p1 - pts[i]
        v2 = p2 - pts[i]
        
        # Calculate the angle between these two vectors
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            continue
            
        unit_v1 = v1 / norm_v1
        unit_v2 = v2 / norm_v2
        
        dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
        angle = np.arccos(dot_product)
        
        # We look for the point whose local neighbors form an angle closest to 90 deg (pi/2)
        angle_diff = abs(angle - np.pi / 2)
        if angle_diff < min_angle_diff:
            min_angle_diff = angle_diff
            best_corner_idx = i

    corner_idx = best_corner_idx
    corner = pts[corner_idx]
    
    # --- 2. Separate the Arms ---
    # Remaining 7 points
    others_mask = np.arange(8) != corner_idx
    others = pts[others_mask]
    other_indices = np.where(others_mask)[0]
    
    # Calculate vectors from corner to all other points
    vectors = others - corner
    
    # Find the point furthest from the corner. This MUST be the tip of the LONG arm.
    farthest_idx_in_others = np.argmax(np.linalg.norm(vectors, axis=1))
    long_arm_end_vec = vectors[farthest_idx_in_others]
    
    # Calculate alignment (cosine similarity) against the long arm vector
    # Points on the long arm will have a similarity close to 1.0; short arm will be near 0.0
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(long_arm_end_vec)
    # Avoid division by zero safely
    norms[norms == 0] = 1e-6
    cos_sim = np.dot(vectors, long_arm_end_vec) / norms
    
    # The 4 points most aligned with the long arm end vector go to the long arm
    long_arm_mask = np.argsort(cos_sim)[-4:]
    short_arm_mask = np.argsort(cos_sim)[:3]
    
    long_arm_pts_indices = other_indices[long_arm_mask]
    short_arm_pts_indices = other_indices[short_arm_mask]
    
    # --- 3. Sort points within arms by distance from corner ---
    long_arm_pts = pts[long_arm_pts_indices]
    dist_long = np.linalg.norm(long_arm_pts - corner, axis=1)
    sorted_long_indices = long_arm_pts_indices[np.argsort(dist_long)]
    
    short_arm_pts = pts[short_arm_pts_indices]
    dist_short = np.linalg.norm(short_arm_pts - corner, axis=1)
    sorted_short_indices = short_arm_pts_indices[np.argsort(dist_short)]
    
    # --- 4. Final Assembly ---
    final_indices = [corner_idx] + list(sorted_long_indices) + list(sorted_short_indices)
    image_points = pts[final_indices]
    
    # 3D object points in cm (or meters, as specified by your coordinates)
    object_points = np.array([
        [0.0,    0.0,    0.230],  # corner
        [0.130,  0.0,    0.0],
        [0.255,  0.0,    0.0],
        [0.380,  0.0,    0.0],
        [0.505,  0.0,    0.0],  # long arm

        [0.0,   0.125,  0.230],
        [0.0,   0.250,  0.230],
        [0.0,   0.375,  0.230],  # short arm
    ], dtype=np.float32)

    info = {
        "corner_index": int(corner_idx),
        "long_arm_indices": sorted_long_indices.tolist(),
        "short_arm_indices": sorted_short_indices.tolist(),
    }

    return image_points.astype(np.float32), object_points, info


def reprojection_error(object_points, image_points, rvec, tvec, K, dist_coeffs):
    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, K, dist_coeffs
    )
    projected = projected.reshape(-1, 2)
    err = np.mean(np.linalg.norm(projected - image_points, axis=1))
    return err, projected


def camera_position_from_pose(Rot, tvec):
    """
    OpenCV pose convention:
        X_cam = R * X_obj + t
    Camera center in object coordinates:
        C_obj = -R^T * t
    """
    return -Rot.T @ tvec

def estimate_pose_nonplanar(object_points, image_points, K, dist_coeffs):
    object_points = np.ascontiguousarray(object_points, dtype=np.float64).reshape(-1, 3)
    image_points = np.ascontiguousarray(image_points, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64)

    if object_points.shape[0] < 4:
        raise ValueError("Need at least 4 points")
    if image_points.shape[0] != object_points.shape[0]:
        raise ValueError("image_points and object_points must match in count")

    # success, rvec, tvec = cv2.solvePnP(
    #     object_points,
    #     image_points,
    #     K,
    #     None,
    #     flags=cv2.SOLVEPNP_EPNP
    # )

    # R_wld_to_cam, _ = cv2.Rodrigues(rvec)
    # if not success:
    #     return  False, False

    success, rvecs, tvecs, reproj_errors = cv2.solvePnPGeneric(
        object_points,
        image_points,
        K,
        dist_coeffs,
        flags=cv2.SOLVEPNP_AP3P
    )

    if not success:
        return  False, False, None, None, None, None

    best_idx = np.argmin(
        [float(err) for err in reproj_errors]
    )

    rvec = rvecs[best_idx]
    tvec = tvecs[best_idx]

    rvec, tvec = cv2.solvePnPRefineLM(
        object_points,
        image_points,
        K,
        dist_coeffs,
        rvec,
        tvec
    )

    err, projected = reprojection_error(
        object_points, image_points, rvec, tvec, K, dist_coeffs
    )
             
    # cam_pos = camera_position_from_pose(R_mat, tvec)
    R_wld_to_cam, _ = cv2.Rodrigues(rvec)

    # Check that all points are in front of the camera
    pts_cam = (R_wld_to_cam @ object_points.T + tvec).T
    positive_depth = np.all(pts_cam[:, 2] > 0)

    return success, positive_depth, err, rvec, tvec, projected

def estimate_planar_pose(object_points, image_points, K, dist_coeffs):
    """
    Estimate pose for coplanar object points.

    Parameters
    ----------
    object_points : (N,3) ndarray
        Coplanar 3D points, usually all Z=0.
    image_points : (N,2) ndarray
        Corresponding pixel coordinates.
    K : (3,3) ndarray
        Camera intrinsic matrix.
    dist_coeffs : ndarray or None
        Distortion coefficients. Set to zeros if unknown.

    Returns
    -------
    result : dict
        Contains pose, reprojection error, and camera position.
    """
    object_points = np.ascontiguousarray(object_points, dtype=np.float64).reshape(-1, 3)

    image_points = np.ascontiguousarray(image_points, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64)

    if object_points.shape[0] < 4:
        raise ValueError("Need at least 4 points")
    if image_points.shape[0] != object_points.shape[0]:
        raise ValueError("image_points and object_points must match in count")

    # IPPE is designed for planar pose estimation.
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        None,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

#    if not success:
#        # Fallback to iterative
#        success, rvec, tvec = cv2.solvePnP(
#            object_points,
#            image_points,
#            K,
#            None,
#            flags=cv2.SOLVEPNP_ITERATIVE
#        )

    if not success:
        return {"success": False}

    # Optional refinement
    #success, rvec, tvec = cv2.solvePnP(
    #    object_points,
    #    image_points,
    #    K,
    #    dist_coeffs,
    #    rvec=rvec,
    #    tvec=tvec,
    #    useExtrinsicGuess=True,
    #    flags=cv2.SOLVEPNP_ITERATIVE
    #)

    R_wld_to_cam, _ = cv2.Rodrigues(rvec)
    #print('tvec', tvec)
    T_wld_to_cam = np.eye(4)
    T_wld_to_cam[:3, :3] = R_wld_to_cam
    T_wld_to_cam[:3, 3] = tvec.flatten()

    # Invert transform
    T_cam_to_wld = np.linalg.inv(T_wld_to_cam)

    # R_mat, _ = cv2.Rodrigues(rvec)
    # R_cam_to_w = R_mat.T  # Camera orientation in world coordinates
    # cam_orient_quat = R.from_matrix(R_cam_to_w).as_quat()  # (x, y, z, w)

    err, projected = reprojection_error(
        object_points, image_points, rvec, tvec, K, None
    )
             
    # cam_pos = camera_position_from_pose(R_mat, tvec)

    # Check that all points are in front of the camera
    pts_cam = (R_wld_to_cam @ object_points.T + tvec).T
    positive_depth = np.all(pts_cam[:, 2] > 0)

    T_cam_to_drone = np.array([
        [ 0, -1,  0, 0],
        [ 1,  0,  0, 0],
        [ 0,  0,  1, 0],
        [ 0,  0,  0, 1],
    ])

    T_drone_to_wld = T_cam_to_wld 
    #T_drone_to_wld = T_cam_to_wld @ np.linalg.inv(T_cam_to_drone)

    cam_pos = T_drone_to_wld[:3, 3]
    cam_orient_quat = R.from_matrix(T_drone_to_wld[:3, :3]).as_quat()  # (x, y, z, w)
    # R_body_to_w = R_cam_to_w @ R_cam_to_body.T
    # R_body_to_w_quat =  R.from_matrix(R_body_to_w).as_quat()  # (x, y, z, w)

    return {
        "success": True,
        "rvec": rvec,
        "tvec": tvec,
        "R": R_wld_to_cam,
        "camera_position": cam_pos,
        "camera_orientation": cam_orient_quat,
        "reprojection_error": err,
        "projected_points": projected,
        "positive_depth": positive_depth,
        "points_camera_frame": pts_cam,
    }


