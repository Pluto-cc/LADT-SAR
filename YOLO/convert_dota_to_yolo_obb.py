import sys
# yolo11源码文件夹
sys.path.append('/home/ubuntu/pycharm_file/ultralytics-main')
 
from ultralytics.data.converter import convert_dota_to_yolo_obb
# 上一步骤中设置的文件夹结构根目录
convert_dota_to_yolo_obb('/home/ubuntu/pycharm_file/ultralytics-main/data')