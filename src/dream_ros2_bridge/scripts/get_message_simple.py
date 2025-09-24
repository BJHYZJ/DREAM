# import zmq
# import json

# def main():
#     context = zmq.Context()
#     socket = context.socket(zmq.SUB)
#     socket.connect("tcp://localhost:4404")
#     socket.setsockopt_string(zmq.SUBSCRIBE, "")
    
#     print("waiting for message...")
    
#     try:
#         while True:
#             message = socket.recv_pyobj()
#             print("收到 message:")
#             print(f"Keys: {list(message.keys())}")
#             print(f"Message: {message}")
#             print("-" * 50)
            
#     except KeyboardInterrupt:
#         print("user interrupt")
#     finally:
#         socket.close()
#         context.term()

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rtabmap_msgs.msg import MapData

class MapDataPrinter(Node):
    def __init__(self):
        super().__init__('mapdata_printer')
        self.create_subscription(MapData, '/rtabmap_ranger_xarm/mapData', self.cb, 10)

    def cb(self, msg):
        self.get_logger().info(f"Total nodes: {len(msg.nodes)}")
        if msg.nodes:
            n = msg.nodes[0]
            self.get_logger().info(f"First node: id={n.id}, stamp={n.stamp}, odom pose={n.pose}")
        rclpy.shutdown()

rclpy.init()
node = MapDataPrinter()
rclpy.spin(node)
