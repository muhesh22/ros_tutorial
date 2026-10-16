import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


class ArucoPositionEstimator(Node):

    def __init__(self):
        super().__init__('aruco_position_estimator')

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.frame_count = 0

        # Gazebo marker plane is 0.4 m wide.
        # Black ArUco square occupies 200/300 of the texture.
        self.marker_length = 0.4 * (200.0 / 300.0)

        self.dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )

        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(
            self.dictionary, self.parameters
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            10
        )

        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.get_logger().info(
            'ArUco position estimator started. '
            'Waiting for camera calibration...'
        )

    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(
            msg.k, dtype=np.float64
        ).reshape(3, 3)

        self.dist_coeffs = np.array(
            msg.d, dtype=np.float64
        )

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8'
            )

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = self.detector.detectMarkers(gray)

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                for i, marker_id in enumerate(ids.flatten()):

                    if marker_id != 1:
                        continue

                    if self.camera_matrix is None:
                        cv2.putText(
                            frame,
                            'Waiting for camera calibration',
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2
                        )
                        continue

                    half = self.marker_length / 2.0

                    # Object points for SOLVEPNP_IPPE_SQUARE.
                    # Keep this required ordering unchanged.
                    object_points = np.array([
                        [-half,  half, 0.0],
                        [ half,  half, 0.0],
                        [ half, -half, 0.0],
                        [-half, -half, 0.0]
                    ], dtype=np.float32)

                    image_points = corners[i].reshape(
                        4, 2
                    ).astype(np.float32)

                    success, rvec, tvec = cv2.solvePnP(
                        object_points,
                        image_points,
                        self.camera_matrix,
                        self.dist_coeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )

                    # Reject failed or invalid pose estimates.
                    if not success:
                        continue

                    if not np.all(np.isfinite(tvec)):
                        continue

                    x, y, z = tvec.flatten()

                    # Reject estimates behind the camera.
                    if z <= 0:
                        continue

                    distance = float(np.linalg.norm(tvec))

                    # Draw estimated marker coordinate axes.
                    cv2.drawFrameAxes(
                        frame,
                        self.camera_matrix,
                        self.dist_coeffs,
                        rvec,
                        tvec,
                        self.marker_length * 0.5
                    )

                    cv2.putText(
                        frame,
                        f'Marker ID: {marker_id}',
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f'Camera X: {x:+.2f} m',
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f'Camera Y: {y:+.2f} m',
                        (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f'Camera Z: {z:.2f} m',
                        (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f'Distance: {distance:.2f} m',
                        (10, 135),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2
                    )

                    self.frame_count += 1

                    if self.frame_count % 10 == 0:
                        self.get_logger().info(
                            f'Marker {marker_id} | '
                            f'Image corners: '
                            f'{image_points.tolist()}'
                        )

                        self.get_logger().info(
                            f'tvec: {tvec.flatten().tolist()} | '
                            f'X={x:+.3f} m, Y={y:+.3f} m, '
                            f'Z={z:.3f} m | '
                            f'Distance={distance:.3f} m'
                        )

            cv2.imshow('ArUco Position Estimation', frame)
            cv2.waitKey(1)

        except Exception as error:
            self.get_logger().error(str(error))


def main(args=None):
    rclpy.init(args=args)
    node = ArucoPositionEstimator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
