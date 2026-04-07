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
        Min-Max normalization: 
        Y is mapped to [0, 1].
        X starts at 0 and scales proportionally.
        """
        pts = []
        indices = []
        if keypoints:
            for i, kp in enumerate(keypoints):
                if i < 17 and kp is not None and kp[2] > 0 and (kp[0] != 0 or kp[1] != 0):
                    pts.append((kp[0], kp[1]))
                    indices.append(i)
        
        # Initialize all to 0.0
        result = {}
        for name in cls.kp_names:
            result[f"{name}_x"] = 0.0
            result[f"{name}_y"] = 0.0
            
        if not pts:
            return result

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        
        min_x = min(xs)
        min_y = min(ys)
        max_y = max(ys)
        
        scale_y = max_y - min_y
        if scale_y == 0:
            scale_y = 1.0
            
        # Fill in active ones
        for i, (px, py) in zip(indices, pts):
            name = cls.kp_names[i]
            result[f"{name}_x"] = (px - min_x) / scale_y
            result[f"{name}_y"] = (py - min_y) / scale_y
            
        return result
