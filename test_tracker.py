import unittest
import object_tracker
from yolov3.utils import Load_Yolo_model
import pandas as pd


class MyTestCase(unittest.TestCase):
    def test_objecttracker(self):
        yolo = Load_Yolo_model()
        result = object_tracker.Object_tracking(yolo, "./IMAGES/test.mp4", "detection.avi", input_size=416, show=True, iou_threshold=0.1, rectangle_colors=(255,0,0), Track_only = ["person"])

        self.assertIsInstance(result, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
