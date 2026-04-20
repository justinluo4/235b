import cv2
import numpy as np

class ArucoDetector:
    def __init__(self):
        self.K = np.array([
            [1698.75,    0,    1115.55],
            [   0,    1695.98,  751.98],
            [   0,       0,       1   ]], dtype=np.float64)

        self.d = np.array([-0.00670872, -0.1481124, -0.00250596, 0.00299921, -1.68711031], dtype=np.float64)

        self.tag_size = 0.02  # 2cm tag as said in file

        # Use original ArUco dictionary
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)

    def find_tags(self, frame):
        """Detect ArUco markers in a frame.

        Args:
            frame in BGR image

        Returns:
            List of (id, T) tuples:
            1. id: integer marker ID
            2. T:  4x4 homogeneous transform of marker relative to camera
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        results = []
        if ids is None:
            return results

        half = self.tag_size / 2.0
        obj_pts = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0],
        ], dtype=np.float64)

        for i, marker_id in enumerate(ids.flatten()):
            img_pts = corners[i][0].astype(np.float64)
            success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, self.K, self.d, flags=cv2.SOLVEPNP_IPPE_SQUARE)

            if not success:
                continue

            R, _ = cv2.Rodrigues(rvec)

            T = np.eye(4)
            T[:3, :3] = R
            T[:3,  3] = tvec.flatten()

            results.append((int(marker_id), T))

        return results



# Test main function
if __name__ == '__main__':
    detector = ArucoDetector()

    frame = cv2.imread('aruco_detection_test_practice.png')
    #frame = cv2.imread('1.png')
    tags = detector.find_tags(frame)

    print(f"Found {len(tags)} tag(s):")
    for tag_id, T in tags:
        print(f"\nID: {tag_id}")
        print(f"  Position (x, y, z): {T[:3, 3]}")
        print(f"  Transform:\n{np.round(T, 4)}")
