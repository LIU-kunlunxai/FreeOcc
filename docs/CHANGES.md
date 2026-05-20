# FreeOcc 修改记录

## 数据流总览

```
ROS2 bag (LiDAR + RGB + Depth)
  │
  ├─ FAST-LIVO2 (C++ SLAM) ──→ poses.txt + per-frame PCDs
  │
  ├─ fastlivo Depth-ICP ──→ 标定好的 T_lidar_cam (外参)
  │
  ├─ preprocess_ros2bag.py ──→ FreeOcc dataset (color/ depth/ pose/)
  │      │                          depth 来自 LiDAR 投影 → 替代反光相机深度
  │      └─ 或用 preprocess_lidar.py 一键完成标定+投影
  │
  └─ run_freeocc.sh ──→ Step1: 高斯建图 → final_rgbd.ply
           │              Step2: 占位投影 → occupancy/ + voxel_features.pkl
           │
           └─ query_occ.py ──→ 在线文本查询 "door" → 3D 坐标 + 热力图
```

## 修改的文件

### 核心修复

| 文件 | 改动 | 原因 |
|------|------|------|
| `src/droid_net.py:4` | 注释 `from torch_scatter import scatter_mean`，改为函数内懒加载 | torch_scatter 与 torch 2.9 不兼容，`use_gt_poses=True` 时不走 DroidNet |
| `src/gaussian_mapping.py:385` | `device = g._surface_xyz.device if hasattr(g, "_surface_xyz") else g._xyz.device` | mono 模式无 `_surface_xyz`，空列表传进 `optimize_scale_rotation` 崩溃 |
| `src/gaussian_mapping.py:1817` | `get_current_gaussians()` 返回值增加是否为空判断 | warmup 阶段高斯列表为空 |
| `src/gaussian_mapping.py:1933` | `get_aligned_gaussians()` 为空时返回 None | 同上 |
| `src/gaussian_mapping.py:1006,1788` | 保存 PLY 前判断 None | 空高斯对象调 save_ply 崩溃 |
| `src/gaussian_mapping.py:1982` | 空返回统一为 `[], {}, []` | 多处返回值不一致 |
| `run.py:175-177` | `output_folder` 非默认值时优先生效 | Hydra 目录不被用户参数覆盖 |
| `run.py:178` | 加 `os.makedirs(output_folder, exist_ok=True)` | `output_folder` 提前写 config.yaml 时目录可能不存在 |

### 开集语义 (CLIP 特征)

| 文件 | 改动 |
|------|------|
| `thirdparty/Trident/trident.py` | 新增 `get_clip_features()` 方法，返回原始 512 维 CLIP 特征（不乘 query_features） |
| `src/gaussian_mapping.py:641` | 新增 `store_clip_features` 参数，为 True 时存 CLIP 特征而非类别 logits |
| `src/gaussian_mapping.py:1337` | 新增 `get_clip_pred()` 方法 |
| `src/gaussian_mapping.py:2019` | `store_clip_features=True` 时调用 `get_clip_pred` |
| `src/gaussian_mapping.py:2032` | CLIP 模式下跳过 argmax label（512 维无意义） |
| `configs/mapping/base.yaml:48` | 新增 `store_clip_features: False` 配置项 |

### 标定迁移 (fastlivo)

| 文件 | 来源 | 说明 |
|------|------|------|
| `scripts/fastlivo/__init__.py` | 新建 | 模块初始化 |
| `scripts/fastlivo/config.py` | fastlivo/pipeline/ | PipelineConfig dataclass |
| `scripts/fastlivo/io_pcd.py` | fastlivo/pipeline/ | PCD 读写 + 体素降采样 |
| `scripts/fastlivo/io_bag.py` | fastlivo/pipeline/ | ROS2 bag 读取 (需改 rosbags) |
| `scripts/fastlivo/poses.py` | fastlivo/pipeline/ | TUM 位姿 + 全局地图 |
| `scripts/fastlivo/calibrate.py` | fastlivo/pipeline/ | Depth-ICP 外参优化 |
| `scripts/fastlivo/colorize.py` | fastlivo/pipeline/ | LiDAR→相机投影着色 |
| `scripts/fastlivo/main.py` | fastlivo/pipeline/ | 5 步管线入口 |

### 新增脚本

| 文件 | 用途 |
|------|------|
| `scripts/run_freeocc.sh` | 一键运行：建图 → 占位投影 → 体素特征 |
| `scripts/preprocess_ros2bag.py` | ROS2 bag → FreeOcc 数据集 (支持 LiDAR 深度投影) |
| `scripts/preprocess_lidar.py` | 完整 LiDAR 预处理 (标定+投影+输出) |
| `scripts/visualize_lidar_projection.py` | 可视化 LiDAR→相机投影对齐效果 |
| `scripts/occ_from_ply.py` | PLY → 占用体素 + 语义俯视图 + 体素特征缓存 |
| `scripts/query_occ.py` | 在线开集语义查询 (文本→3D坐标) |
| `scripts/merge_sessions.py` | 多段 PLY 融合到统一体素 |
| `scripts/fuse_lidar_occ.py` | LiDAR 几何 + FreeOcc 语义融合 |
| `scripts/semantic_ply_viewer.py` | PLY 语义着色子采样 (3D查看器用，CLIP模式自适应) |
| `scripts/setup_remote.sh` | 远程服务器环境安装 |

### 配置与数据

| 文件 | 说明 |
|------|------|
| `scripts/fastlivo/config_calibrated.yaml` | 标定好的外参 (Depth-ICP 优化结果) |
| `src/scannet_utils/kunlunxai_name.txt` | 自定义语义类别 (用户场景) |
| `src/scannet_utils/scannet_name.txt` | ScanNet 11 类 (默认) |
| `src/scannet_utils/realsense_name.txt` | RealSense 20 类 |
| `src/scannet_utils/replica_name.txt` | Replica 100 类 |

## 关键配置参数

```bash
# 模式选择
mode=rgbd                      # 必须有传感器深度 (LiDAR投影或用传感器)
use_gt_poses=True              # 跳过 SLAM，用外部 LiDAR 位姿
sync_method=relaxed            # 降低多进程 CUDA 同步冲突

# 开集语义
mapping.store_clip_features=True              # 存 512 维 CLIP 特征
mapping.ov_name_path=./src/scannet_utils/kunlunxai_name.txt  # 类别列表

# 深度过滤 (对抗反光地面)
mapping.loss.supervise_with_prior=False       # 用多视图过滤后的深度
mapping.online_opt.filter.bin_th=0.02        # 多视图一致性阈值
mapping.online_opt.filter.uncertainty=True    # 不确定性过滤
mapping.online_opt.filter.conf_th=0.05       # 置信度过滤

# 数据路径
data.input_folder=/path/to/dataset            # color/ depth/ pose/
data.cam.H=480 data.cam.W=640                 # 相机分辨率
data.cam.fx=386.502 data.cam.fy=385.938       # 内参
data.cam.cx=321.497 data.cam.cy=241.840
data.png_depth_scale=1000.0                   # 深度图单位 (mm→m)

# 可视化
run_visualization=False                       # 无显示器必须关
run_mapping_gui=False                         # 无显示器必须关
mapping.enable_occ_eval=False                 # 无 GT 占用数据必须关
```

## 数据集格式

```
dataset/
  color/          0.jpg, 1.jpg, ...      ← OpenCV 可读的 RGB 图像
  depth/          0.png, 1.png, ...      ← 16-bit PNG 深度图 (毫米)
  pose/           0.txt, 1.txt, ...      ← 4×4 camera-to-world 矩阵
  depth_lidar/    0.png, ...             ← (可选) LiDAR 投影深度
                                            run_freeocc.sh 自动检测
```

## 输出目录结构

```
freeocc_output/2026-05-19_21-29-58/
  mesh/
    final_rgbd.ply                           ← 738 万高斯 (512 维 CLIP 特征)
  occupancy/
    occ_voxel.ply                            ← 3D 占用体素 (密度着色)
    occ_voxel_sem_label.ply                  ← 3D 语义体素 (类别着色 - 仅 logit 模式)
    occ_topdown_zmax.png                     ← XY 俯视图 (密度)
    occ_topdown_sem_label.png                ← XY 俯视图 (语义 - 仅 logit 模式)
    legend.png                               ← 颜色图例
    voxel_features.pkl                       ← 体素级 CLIP 特征 (开集查询用)
```

## 已知问题

| 问题 | 状态 | 解决 |
|------|------|------|
| torch_scatter 与 torch 2.9 不兼容 | 已绕过 | 懒加载 import |
| mono 模式 + GT 位姿无深度 | 不可行 | 必须用 rgbd 模式 |
| open3d 可视化线程无头崩溃 | 已关闭 | `run_visualization=False` |
| Hydra 不接受新 key | 已修复 | base.yaml 里注册或加 `+`/`++` |
| CUDA IPC 多进程崩溃 | 缓解 | `sync_method=relaxed` + PYTORCH_ALLOC_CONF |
| plyfile v.count→len(v) | 已适配 | 新版本 API 变更 |
| 相机深度反光地面不准 | 已解决 | LiDAR 投影深度替代 |

## 待完成

- [ ] `scripts/fastlivo/` 内相对导入改为绝对导入
- [ ] `scripts/fastlivo/io_bag.py` rosbag2_py → rosbags 适配
- [ ] `merge_sessions.py` 加入 recency-based 融合 (新 session 覆盖旧)
- [ ] `gaussian_mapping.py` 存储 CLIP 特征时直接存 strided feature (减少维度)
