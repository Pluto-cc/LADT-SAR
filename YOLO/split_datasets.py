import os
import random
import shutil
 
random.seed(42)
 
"""
该脚本用于将给定的数据集分割成训练集和测试集。
数据集应包含图像和对应的标注文件。
脚本会按照90%训练集和10%测试集的比例进行分割，并将图像和标注文件分别复制到相应的文件夹中。
"""
 
# 设置数据集文件夹路径和输出文件夹路径
data_folder = './data_mouse_ro'
img_folder = './data_mouse_ro/dataset/images'
label_folder = './data_mouse_ro/dataset/labels'
 
# 计算每个子集的大小
# 总文件数乘以0.9得到训练集大小，其余为测试集大小
total_files = len(os.listdir(os.path.join(data_folder, 'img')))
train_size = int(total_files * 0.9)
test_size = int(total_files - train_size)
 
# 获取所有图像文件的文件名列表，并进行随机打乱
image_files = os.listdir(os.path.join(data_folder, 'img'))
random.shuffle(image_files)
 
# 复制图像和标注文件到相应的子集文件夹中
# 枚举每个图像文件，根据索引决定复制到训练集还是测试集文件夹
for i, image_file in enumerate(image_files):
    base_file_name = os.path.splitext(image_file)[0]  # 获取文件名（不包括扩展名）
    image_path = os.path.join(data_folder, 'img', image_file)
    label_path = os.path.join(data_folder, 'dotatxt', base_file_name + '.txt')
 
    # 根据索引判断文件应复制到训练集还是测试集
    if i < train_size:
        shutil.copy(image_path, os.path.join(img_folder, 'train'))  # 复制图像到训练集
        shutil.copy(label_path, os.path.join(label_folder, 'train_original'))  # 复制标注到训练集
    else:
        shutil.copy(image_path, os.path.join(img_folder, 'val'))  # 复制图像到测试集
        shutil.copy(label_path, os.path.join(label_folder, 'val_original'))  # 复制标注到测试集