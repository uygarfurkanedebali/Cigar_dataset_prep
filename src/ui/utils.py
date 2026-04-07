import math

class PoseNormalizer:
    kp_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    @classmethod
    def normalize_keypoints(cls, keypoints):
        """
        1. Alignment (Rotation around mid_hip if shoulders/hips exist)
        2. Min-Max normalization: Y is mapped to [0, 1], X scales proportionally.
        """
        # Ensure keypoints exist and extract active ones
        all_pts = [None] * 17
        if keypoints:
            for i, kp in enumerate(keypoints):
                if i < 17 and kp is not None and kp[2] > 0 and (kp[0] != 0 or kp[1] != 0):
                    all_pts[i] = (kp[0], kp[1])

        # Initialize result with zeros
        result = {f"{name}_{axis}": 0.0 for name in cls.kp_names for axis in ['x', 'y']}

        # Check for rotation requirements (Shoulders 5,6 and Hips 11,12)
        k5, k6, k11, k12 = all_pts[5], all_pts[6], all_pts[11], all_pts[12]
        should_rotate = all(p is not None for p in [k5, k6, k11, k12])

        rotated_pts = {} # index -> (x, y)
        if should_rotate:
            mid_sh = ((k5[0] + k6[0])/2, (k5[1] + k6[1])/2)
            mid_hp = ((k11[0] + k12[0])/2, (k11[1] + k12[1])/2)
            
            # Spine vector: from mid_hip to mid_shoulder
            # Target angle: UP in image coordinates (-math.pi / 2)
            current_angle = math.atan2(mid_sh[1] - mid_hp[1], mid_sh[0] - mid_hp[0])
            target_angle = -math.pi / 2
            theta = target_angle - current_angle
            
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            
            # Rotate all points around mid_hp
            for i, p in enumerate(all_pts):
                if p is not None:
                    dx, dy = p[0] - mid_hp[0], p[1] - mid_hp[1]
                    rx = mid_hp[0] + dx * cos_t - dy * sin_t
                    ry = mid_hp[1] + dx * sin_t + dy * cos_t
                    rotated_pts[i] = (rx, ry)
        else:
            # Skip rotation and use original points
            for i, p in enumerate(all_pts):
                if p is not None:
                    rotated_pts[i] = p

        if not rotated_pts:
            return result

        # Apply Min-Max on rotated coordinates
        xs = [p[0] for p in rotated_pts.values()]
        ys = [p[1] for p in rotated_pts.values()]
        
        min_x = min(xs)
        min_y = min(ys)
        max_y = max(ys)
        
        scale_y = max_y - min_y
        if scale_y == 0:
            scale_y = 1.0
            
        for i, (rx, ry) in rotated_pts.items():
            name = cls.kp_names[i]
            result[f"{name}_x"] = (rx - min_x) / scale_y
            result[f"{name}_y"] = (ry - min_y) / scale_y
            
        return result
