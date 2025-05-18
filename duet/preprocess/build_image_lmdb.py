import os
import sys
import numpy as np
import json
import collections
from PIL import Image
import matplotlib.pyplot as plt
import lmdb
import math
import imageio

# sys.path.append('/projects/VLN-Tutorial/Matterport3DSimulator/build')
import MatterSim

# 如果tmp文件夹不存在，则创建
# 创建临时目录用于存储调试图像
if not os.path.exists('tmp'):
    os.makedirs('tmp')

# 模拟器图像参数设置
# 设置图像的宽度、高度和垂直视场角
WIDTH = 640      # 原始图像宽度
HEIGHT = 480     # 原始图像高度
VFOV = 60        # 垂直视场角（度）
DEBUG = True    # 调试模式开关

# 数据路径设置
# Matterport3D数据集路径和连接图路径
scan_data_dir = '/projects/VLN-Tutorial/duet/datasets/Matterport3D/v1_unzip_scans'
connectivity_dir = '/projects/VLN-Tutorial/duet/datasets/R2R/connectivity'

# 初始化MatterSim模拟器
# 这个模拟器用于在3D环境中捕获全景图像
sim = MatterSim.Simulator()
sim.setDatasetPath(scan_data_dir)
sim.setNavGraphPath(connectivity_dir)
# 当启用预加载时，所有全景图像会在开始前加载到内存中。预加载需要几分钟时间，并且需要约50GB内存(RGB输出)或80GB(启用深度输出)，但渲染速度会快得多。
# sim.setPreloadingEnabled(True)
sim.setCameraResolution(WIDTH, HEIGHT)
sim.setCameraVFOV(math.radians(VFOV))
sim.setDiscretizedViewingAngles(True)
sim.setBatchSize(1)
sim.initialize()

# 读取所有视点ID
# 从连接图文件中获取所有可用的视点
viewpoint_ids = []
with open(os.path.join(connectivity_dir, 'scans.txt')) as f:
    scans = [x.strip() for x in f]
for scan in scans:
    with open(os.path.join(connectivity_dir, '%s_connectivity.json'%scan)) as f:
        data = json.load(f)
        viewpoint_ids.extend([(scan, x['image_id']) for x in data if x['included']])
print('Loaded %d viewpoints' % len(viewpoint_ids))  # 输出加载的视点数量


# 设置生成图像的新尺寸
# 为了减小存储空间，将原始图像调整为较小的尺寸
NEWHEIGHT = 248
NEWWIDTH = int(WIDTH / HEIGHT * NEWHEIGHT)
print(NEWHEIGHT, NEWWIDTH)

# 计算每张图像的大小和总数据大小
# 估算LMDB数据库需要的存储空间
data_size_per_img = np.random.randint(255, size=(NEWHEIGHT, NEWWIDTH, 3), dtype=np.uint8).nbytes
print(data_size_per_img, 36*data_size_per_img*len(viewpoint_ids))

# 设置LMDB数据库路径
# LMDB是一种高性能的键值存储数据库，用于存储全景图像
lmdb_path = 'duet/datasets/R2R/features/panoimages_test.lmdb'

# 创建LMDB环境
# map_size设置为足够大的值以容纳所有图像数据
env = lmdb.open(lmdb_path, map_size=int(1e12))


# 遍历所有视点，为每个视点生成全景图像
# 每个视点捕获36个不同角度的图像，组成一个完整的全景
for i, viewpoint_id in enumerate(viewpoint_ids):
    scan, vp = viewpoint_id
    if i % 100 == 0:
        print(i, scan, vp)  # 每处理100个视点输出一次进度
    
    # 生成键名，格式为"扫描ID_视点ID"
    key = '%s_%s' % (scan, vp)
    key_byte = key.encode('ascii')  # 转换为字节类型用于LMDB
    
    # 开始LMDB事务
    txn = env.begin(write=True)
        
    # 捕获36个视角的图像
    images = []
    for ix in range(36):
        if ix == 0:
            # sim.newEpisode函数：开始一个新的模拟场景（episode）
            # 参数解释：
            # 1. scan列表：指定3D扫描场景的ID
            # 2. vp列表：指定起始视点(viewpoint)的ID
            # 3. [0]：初始水平朝向角度（弧度），0表示初始y轴正方向，向右转为正
            # 4. [math.radians(-30)]：初始摄像机仰角（弧度），表示从水平面向下倾斜30度，上为正，下为负
            sim.newEpisode([scan], [vp], [0], [math.radians(-30)])
        elif ix % 12 == 0:
            # sim.makeAction函数：在模拟环境中执行动作
            # 参数解释：
            # 1. [0]：导航位置索引，0表示保持在当前位置不移动（不改变视点）
            # 2. [1.0]：水平旋转角度（弧度），约57.3度，正值表示向右转
            # 3. [1.0]：垂直旋转角度（弧度），约57.3度，正值表示向上看
            # 这里执行了改变垂直视角的动作，向上看
            sim.makeAction([0], [1.0], [1.0])
        else:
            # 这里执行了水平旋转的动作，保持垂直角度不变
            # 1. [0]：保持在当前位置
            # 2. [1.0]：水平旋转约57.3度（每次旋转30度）
            # 3. [0]：垂直角度不变
            sim.makeAction([0], [1.0], [0])
        
        #此处参考mattersim文档了解sim接口
        #https://github.com/peteanderson80/Matterport3DSimulator/tree/589d091b111333f9e9f9d6cfd021b2eb68435925?tab=readme-ov-file#simulator-api

        # 获取当前状态和图像
        state = sim.getState()[0]
        assert state.viewIndex == ix
        image = np.array(state.rgb, copy=True)  # BGR格式的图像
        image = Image.fromarray(image[:, :, ::-1])  # 转换为RGB格式
        
        # 调整图像大小以节省存储空间
        image = image.resize((NEWWIDTH, NEWHEIGHT), Image.Resampling.LANCZOS)
        image = np.array(image)
        images.append(image)
        
        # 如果开启调试模式，保存单独的图像文件
        if DEBUG:   
            save_path = os.path.join('tmp', f'{scan}_{vp}_{ix}.png')
            imageio.imwrite(save_path, image)
    
    # 将所有36个视角的图像堆叠为一个数组
    images = np.stack(images, 0)
    
    # 将图像数据存储到LMDB数据库
    txn.put(key_byte, images)
    txn.commit()

# 关闭LMDB环境
env.close()

# 思考： 如何读取指定key的数据？
# 答： 使用txn.get(key_byte)函数读取指定key的数据
# 读取指定key的数据
# with env.begin() as txn:
#     data = txn.get(key_byte)
#     print(data)
