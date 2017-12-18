# def super_parameters():
# coco config parameters


coco = {"keypoints": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
                      "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                      "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"],
        "skeleton": [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
                     [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]}

coco_openpose = {"skeleton": [[6, 8], [8, 10], [7, 9], [9, 11], [12, 14], [14, 16], [13, 15], [15, 17], [0, 1], [0, 6],
                              [0, 7], [0, 12], [0, 13], [2, 4], [5, 3], [1, 2], [1, 3], [4, 6], [5, 7]],
                 "keypoints": ["neck", "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
                               "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                               "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]}
