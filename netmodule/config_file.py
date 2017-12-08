
#def super_parameters():
#coco config parameters
coco18 = {'keypoints': {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist", 5: "LShoulder",
                      6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee", 10: "RAnkle", 11: "LHip",
                      12: "LKnee", 13: "LAnkle", 14: "REye", 15: "LEye", 16: "REar", 17: "LEar", 18: "Bkg"},
          'limbSequence': [1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 1, 8, 8, 9, 9, 10, 1, 11, 11, 12,
                         12, 13, 1, 0, 0, 14, 14, 16, 0, 15, 15, 17, 2, 16, 5, 17]}
coco={"keypoints": ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder",
                    "right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
                    "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"],
	"skeleton": [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],
                 [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]}


