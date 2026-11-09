# Z01 机器人仿真资产

**Z01 人形机器人**（Zvalley Robotics）的 MuJoCo / URDF 仿真资产库，包含基线模型及 RoboCup 类人组开发所用的各变体。

[English](README.md) | 中文

## 模型变体

| 目录 | 可动关节 | 说明 |
| --- | --- | --- |
| `Z01_std` | 29 | 基线模型：双臂 2 × 7 自由度（肩 ×3、肘、腕 ×3）、腰 3 自由度、双腿 2 × 6 自由度（髋 ×3、膝、踝 ×2） |
| `Z01_std_add_head` | 29 | 在 `Z01_std` 基础上增加 2 自由度头部（`head_pitch` / `head_yaw`）与相机挂载；腰 pitch/roll 焊死 |
| `Z01_std_parallel_ankle` | 33 | 增加踝部并联机构：每侧两根连杆关节（`ankle_1` / `ankle_2`）通过肌腱 equality 约束与 `ankle_pitch` / `ankle_roll` 刚性耦合。**多机器人场景中机器人本体的唯一真源。** |
| `Z01_std_football_player` | 33 | 机器人本体同 `parallel_ankle`，另含 RoboCup 足球场场景（`xml/scene/soccer_field.xml`、`soccer_field_assets.xml`、`center_circle_ring.obj`） |
| `Z01_std_robocup` | 6 × 33 | 生成的多机器人场景：同一片场地上 6 台 `parallel_ankle` 机器人，各带一路深度相机。详见 [`Z01_std_robocup/README.md`](Z01_std_robocup/README.md) |

## 仓库结构

```
zv_robot_assets/
├── Z01_std/                     # 基线模型，29 自由度
├── Z01_std_add_head/            # + 2 自由度头部 + 相机
├── Z01_std_parallel_ankle/      # + 踝部并联机构（肌腱耦合）
├── Z01_std_football_player/     # + RoboCup 足球场场景
└── Z01_std_robocup/             # 6 机器人多机场景（生成物）
```

每个机器人模型目录结构一致：

```
<variant>/
├── urdf/            # URDF 源模型 —— 唯一真源
├── xml/             # 生成的 MuJoCo MJCF 模型（生成物，勿手改）
├── meshes/          # STL/OBJ 网格（毫米单位，导入时缩放 0.001）
├── csv/             # 关节/连杆参数表：限位、力矩、armature、惯量、网格位姿
└── urdf_to_xml.yaml # 生成配置：基座高度、分模式 armature、电机规则、XPath 装饰补丁
```

## 生成流程

`xml/` 下的文件全部是**生成物**。修改机器人的正确方式：

1. 修改 **URDF**（几何、运动学、惯量）、**CSV**（手工实测参数，如 `armature`）或 **`urdf_to_xml.yaml`**（基座高度、分模式 armature、电机 `ctrlrange` 规则、经 XPath 追加的装饰调试几何）。
2. 用配套仿真工作区（`robot_assets/utils/`）的转换器重新生成 MJCF。不要手改 `xml/` 下的任何文件。

`Z01_std_robocup` 场景由 `robot_assets/utils/gen_robocup_multi_xml.py` 生成：

```bash
python3 robot_assets/utils/gen_robocup_multi_xml.py          # 重新生成（主文件 + 6 个机器人片段）
python3 robot_assets/utils/gen_robocup_multi_xml.py --check  # 只校验（重名 / 关节地址 / frustum / 加载）
```

## RoboCup 场景关键数字

由 `gen_robocup_multi_xml.py --check` 校验：

```
nq=247 nv=240 nu=174 ncam=6 ngeom=392
robot_01..06 基座 free joint qpos 偏移：0 / 40 / 80 / 120 / 160 / 200
足球 free joint：                        qpos = 240
```

- 每台机器人占 40 个 qpos（7 浮动基座 + 33 关节）；仿真代码一律**按名字查地址**，不假设偏移。
- 场地（RoboCup 成人组）：14 m × 9 m，球门宽 2.6 m、高 1.2 m，中圈半径 1.8 m；足球置于场心 `(0, 0, 0.11)`。
- 6 台人形互相碰撞必须放大 `<size njmax="4000" nconmax="2000"/>`（实测约 340 个接触）；arena 约 900 MB **虚拟**内存，但 RSS 仅约 95 MB。
- 兼容 MuJoCo **3.1.1**：6 台机器人通过 `<include>` 合并各自完整子模型实现复用（`<attach>` / `<replicate>` 需要 3.2+ / 3.4+）。

## 下游仿真包注意事项

- `mj_sim` 通过 `file(GLOB */xml/*.xml)` 收集模型；新增模型后需执行 `colcon build --cmake-force-configure --packages-select mj_sim` 重新生成 `robot_header/<model>.h`。
- 设计说明（为何拆成 6 个 `<include>` 子模型、深度相机 frustum、arena 内存表现、header 生成器注意事项）见 [`Z01_std_robocup/README.md`](Z01_std_robocup/README.md)。
