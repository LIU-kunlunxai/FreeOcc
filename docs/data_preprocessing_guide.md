# FreeOcc 数据预处理指南

## 背景

FreeOcc 需要的数据格式：

```
sequence/
  color/    0.jpg, 1.jpg, ...    # RGB 图像
  depth/    0.png, 1.png, ...    # 16-bit PNG 深度图 (毫米)
  pose/     0.txt, 1.txt, ...    # 4x4 camera-to-world 矩阵
```

数据来源：ROS2 bag + LiDAR SLAM 位姿 + 外参标定文件。

---

## 数据文件说明

| 文件 | 位置 | 内容 |
|------|------|------|
| ROS2 bag (`.db3`) | 录制的数据目录 | RGB 图像、深度图、相机内参、LiDAR 点云、IMU |
| `poses_lidar.txt` | `output/` | LiDAR SLAM 位姿，每行 `timestamp tx ty tz qx qy qz qw` |
| `tfs.json` | `output/` | LiDAR→相机外参，`from: livox_frame, to: camera_body_color_optical_frame` |
| `map.pcd` | `output/` | LiDAR SLAM 点云地图 |

### ROS2 Bag 关键 Topic

| Topic | 类型 | 用途 |
|-------|------|------|
| `/camera/body/color/image_raw` | `sensor_msgs/Image` | RGB 图像 (body camera, 77帧) |
| `/camera/body/depth/image_raw` | `sensor_msgs/Image` (16UC1) | 深度图 (body camera) |
| `/camera/body/color/camera_info` | `sensor_msgs/CameraInfo` | 相机内参 |
| `/camera/head/color/image_raw` | `sensor_msgs/Image` | RGB 图像 (head camera, 81帧) |
| `/livox/lidar` | `sensor_msgs/PointCloud2` | LiDAR 点云 (428次扫描) |
| `/tf_static` | `tf2_msgs/TFMessage` | 静态 TF 变换 (外参) |

---

## 预处理脚本

### `scripts/preprocess_ros2bag.py`

从 ROS2 bag + LiDAR SLAM 生成 FreeOcc 格式数据集。

```bash
conda run -n vggt-slam python scripts/preprocess_ros2bag.py \
    --bag_dir /path/to/bag \
    --output_dir /path/to/freeocc_input \
    --camera body \
    --project_lidar_depth \
    --time_window_ms 200
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--bag_dir` | (必需) | ROS2 bag 目录路径 |
| `--output_dir` | (必需) | 输出 FreeOcc 数据集目录 |
| `--camera` | `body` | 相机选择 (body/head)，body 有外参标定 |
| `--poses_file` | `{bag_dir}/output/poses_lidar.txt` | LiDAR SLAM 位姿文件 |
| `--extrinsic_file` | `{bag_dir}/output/tfs.json` | 外参文件 |
| `--project_lidar_depth` | 关闭 | 用 LiDAR 点云投影生成深度图 |
| `--time_window_ms` | `200` | LiDAR 扫描累积窗口 (毫秒) |
| `--max_pose_dt_ms` | `100` | 位姿匹配最大时间差 |

**深度图策略：**

| 方式 | 命令 | 深度来源 | 特点 |
|------|------|---------|------|
| LiDAR 投影 | `--project_lidar_depth` | LiDAR 点云投影 | 稀疏但几何准确 (~8000像素/帧) |
| 相机深度 | 不加参数 | 相机原始深度 | 密集 (~280k像素/帧) 但反光区域不准 |

### 坐标转换流程

```
LiDAR SLAM 位姿                    外参 (ROS TF)
T_world_lidar (c2w)          ×     T_camera_to_livox     =     T_world_camera
[poses_lidar.txt]                  [tfs.json]                   [pose/{i}.txt 输出]
```

ROS TF 约定：`livox_frame → camera_body_color_optical_frame` 表示从 `camera_frame` 到 `livox_frame` 的变换。投影时取逆矩阵得到 `livox→camera`。

---

## 可视化工具

### `scripts/visualize_lidar_projection.py`

生成所有帧的 4 列对比图，验证外参是否正确。

```bash
conda run -n vggt-slam python scripts/visualize_lidar_projection.py \
    --bag_dir /path/to/bag \
    --output_dir /path/to/output_frames \
    --time_window_ms 200 \
    --stride 5
```

**输出 (每帧一张 4 列 PNG)：**

```
┌────────┬────────────────────┬──────────────────┬──────────────────┐
│  RGB   │ LiDAR overlay      │ LiDAR depth      │ Camera depth     │
│        │ (红=近→绿→蓝=远)    │ (灰度: 近白远黑)   │ (灰度: 近白远黑)   │
└────────┴────────────────────┴──────────────────┴──────────────────┘
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--bag_dir` | (必需) | ROS2 bag 目录 |
| `--output_dir` | (必需) | 输出图片目录 |
| `--camera` | `body` | 相机选择 |
| `--stride` | `1` | 每隔 N 帧生成一张 |
| `--start` / `--end` | 0 / 全部 | 帧范围 |
| `--time_window_ms` | `200` | LiDAR 累积窗口 |

**如何判断外参是否正确：** 看第 2 列 (LiDAR overlay)，LiDAR 投影点是否准确落在 RGB 图像中对应物体上。

### 已知问题

- **Camera depth 时间戳不重合**：body camera 的 color 和 depth topic 时间戳差 76~428ms，脚本使用最近邻匹配 (`_find_nearest_cam_depth`)

---

## 运行 FreeOcc

预处理完成后，运行：

```bash
cd /home/liu/workspace/FreeOcc

python run.py \
    mode=rgbd \
    use_gt_poses=True \
    data.input_folder=/path/to/freeocc_input \
    data.cam.H=480 \
    data.cam.W=640 \
    data.cam.H_out=480 \
    data.cam.W_out=640 \
    data.cam.fx=386.502 \
    data.cam.fy=385.938 \
    data.cam.cx=321.497 \
    data.cam.cy=241.840 \
    data.png_depth_scale=1000.0 \
    mapping.online_opt.filter.bin_th=0.03 \
    mapping.loss.supervise_with_prior=False
```

**关键参数说明：**

| 参数 | 含义 |
|------|------|
| `use_gt_poses=True` | 跳过内部 DROID-SLAM，直接使用外部位姿 |
| `mode=rgbd` | 使用 RGB + 深度 |
| `png_depth_scale=1000.0` | 16-bit PNG 深度值 = 毫米 × 1000 |
| `supervise_with_prior=False` | 使用多视图优化后的深度做监督 (而非原始传感器深度) |
| `filter.bin_th=0.03` | 多视图深度一致性阈值 (米) |

### FreeOcc 抗噪声深度机制

1. **多视图一致性过滤** (`filter_map`)：要求同一像素在多个视图中深度一致，否则丢弃
2. **Huber 深度损失**：对离群点不敏感
3. **`supervise_with_prior`**：控制深度监督来源 (原始传感器 vs 多视图优化后)

---

## 依赖环境

- `rosbags`：纯 Python ROS2 bag 解析
- `numpy`, `opencv-python`：数据处理
- 运行环境：`vggt-slam` conda env (含 cv2)

```bash
conda activate vggt-slam
pip install rosbags
```
