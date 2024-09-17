from PIL import Image
import os

def combine_images_horizontal(folder_path, output_filename):
    
    # 获取文件夹下所有以_rgb.png结尾的图片文件
    image_files = [f for f in os.listdir(folder_path) if f.endswith("_0001_rgb.png")]

    # 按文件名排序图片文件，确保拼接的顺序正确
    image_files.sort()

    # 打开第一张图片，作为基准图片
    base_image = Image.open(os.path.join(folder_path, image_files[0]))

    # 获取基准图片的宽度和高度
    base_width, base_height = base_image.size

    # 创建一个新的空白图片，宽度为所有图片的宽度之和，高度为基准图片的高度
    new_image = Image.new('RGB', (base_width * len(image_files), base_height))

    # 将每张图片拼接到新图片上
    x_offset = 0
    for image_file in image_files:
        img = Image.open(os.path.join(folder_path, image_file))
        new_image.paste(img, (x_offset, 0))
        x_offset += base_width

    # 保存新图片
    new_image.save(output_filename)

# 调用函数进行图片拼接，输出为combined_image.png
combine_images_horizontal("/home/lyk/stable-dreamfusion/trial_image_teddy/validation", "/home/lyk/stable-dreamfusion/trial_image_teddy/results/combined_image_1.png")
