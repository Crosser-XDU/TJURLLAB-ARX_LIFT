import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class SaveOneFrame(Node):
    def __init__(self):
        super().__init__('save_one_frame')

        # 用于 ROS Image ↔ OpenCV
        self.bridge = CvBridge()

        # 只保存一次的标志
        self.saved = False

        # 创建订阅者
        self.sub = self.create_subscription(
            Image,
            '/camera/camera_l/color/image_rect_raw',  
            self.cb,
            10
        )

    def cb(self, msg):
        # 如果已经保存过，就什么也不做
        if self.saved:
            return

        # ROS Image → OpenCV
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # 保存图片
        cv2.imwrite('camera_l_one_frame.png', img)

        self.get_logger().info('Saved one frame from camera_l')

        # 标记已保存
        self.saved = True


def main():
    rclpy.init()
    node = SaveOneFrame()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()