# Z01_std_robocup

RoboCup 多机器人场景模型：**同一片足球场上放 6 台 Z01_std_parallel_ankle 机器人**，
供 `src/robot_sim/sim_z01_std_robocup` 加载（每个 MuJoCo 模型 6 台机器人、
6 路深度相机、6 套 `/robot_0X/*` 话题）。

## 生成物，不要手改

```
xml/Z01_std_robocup.xml        主文件（约 120 行）：全局设置 + 6 行 <include> + 场地/光源/足球
xml/robots/robot_0X.xml        6 台机器人各自的**完整子模型**（约 270 行/个）：本体 + actuator/sensor/tendon/equality
```

两份都由生成器产出，请勿手改：

```bash
python3 robot_assets/utils/gen_robocup_multi_xml.py          # 重新生成（主文件 + 6 个片段）
python3 robot_assets/utils/gen_robocup_multi_xml.py --check  # 只校验（重名/关节地址/frustum/加载）
```

**为什么拆成 6 个子模型 include**：MuJoCo 3.1.1 对重名对象是硬报错，而 3.1.1 没有任何
"带改名复用"的 MJCF 机制（`<attach>`/`<replicate>` 是 3.2+/3.4+ 的能力，实测 3.1.1 报
`unrecognized element`），所以 6 台必须各自拥有带 `robot_0X_` 前缀的名字。把每台放在
独立的 `robots/robot_0X.xml`（根标签 `<mujoco>`，**完整子模型**而非 body 片段）后：
`<include>` 会把它的 worldbody / actuator / sensor / tendon / equality 各自合并到主模型
对应段里（3.1.1 实测支持，且 body、执行器、传感器按 include 顺序排列、qpos 顺序 = 文档顺序），
主 XML 因此只剩 120 行，不再有 6 段近乎重复的描述。机器人的唯一真源还是
`Z01_std_parallel_ankle/xml/Z01_std_parallel_ankle.xml`：改一次 + 重跑生成器即可同步 6 台。

## 模型构成

| 部分 | 来源 |
| --- | --- |
| 机器人本体 | `Z01_std_parallel_ankle/xml/Z01_std_parallel_ankle.xml`，生成到 `xml/robots/robot_0X.xml`（对象名加 `robot_0X_` 前缀，含 site/tendon/equality 引用与各台的 actuator/sensor） |
| 足球场（草坪/白线/双方球门/足球/放置预览点） | `Z01_std_football_player/xml/scene/soccer_field.xml`（**内联**，便于把足球放到场心） |
| 纹理/材质/中圈网格 | `<include file="../../Z01_std_football_player/xml/scene/soccer_field_assets.xml"/>` |
| 网格文件 | `<compiler meshdir="../../Z01_std_football_player/meshes/"/>`（该目录同时含机器人 STL 与 `center_circle_ring.obj`，零网格重复） |

与 `Z01_std_football_player` 的两处故意差异：

1. **足球位于场心 `(0, 0, 0.11)`**。football_player 把球藏在 `z=-100`，由它的
   "足球模式"显隐控制；RoboCup 场景里球常显，所以内联场地时改到场心。
2. **放置预览点 `ball_preview` 藏在地面下方 1m**（而不是 `z=-100`）。原来藏在
   -100 会把模型包围盒撑到 107m（`stat.extent`），进而放大深度相机的远近平面；
   放到 -1m 后 `stat.extent≈23`，与场地尺度一致。

## 关键数字（`--check` 输出）

```
nq=247 nv=240 nu=174 ncam=6 ngeom=392
robot_01..06 基座 free joint: qpos = 0 / 40 / 80 / 120 / 160 / 200
足球 free joint:              qpos = 240
```

- 单台机器人 40 个 qpos（7 浮动基座 + 33 转动关节），6 台连续占据 qpos 前段，
  足球在最后 —— 仿真侧一律按**名字查地址**，不假设偏移。
- `<size njmax="4000" nconmax="2000"/>`：6 台人形互相碰撞时实测 `ncon≈340`、
  `nefc≈1600`（单机约 57 / 277），必须放大；arena 因此是 ~900MB **虚拟**内存，
  但 RSS 只有 ~95MB（实测 3.1.1 库、1000 步后 `maxuse_arena≈2.1MB`）。

## 重新生成的前提

- 机器人本体若改动，改的是 `Z01_std_parallel_ankle`（或它的 `urdf_to_xml.yaml`），
  然后重跑本生成器。
- 足球场若改动（`soccer_field.xml`），本模型需要重跑生成器才会同步（场地是内联的，
  不是 include）。
- 新增 XML 后 `mj_sim` 的 `file(GLOB */xml/*.xml)` 需要重新 configure，否则不会
  生成 `robot_header/Z01_std_robocup.h`：
  `colcon build --cmake-force-configure --packages-select mj_sim`。
- 注意 `header_generator.py` 只取**第一个** `<actuator>`/`<sensor>` 块与第一个带
  free joint 的 body，所以生成的 `Z01_std_robocup.h` 里的名字都是 robot_01 的；
  仿真节点**不使用**这个头文件取名字，而是用 `robot_0X_` 前缀 + `Z01_std_parallel_ankle.h`
  拼接（见 `sim_z01_std_robocup` 的 `RobotModelSpec`）。
