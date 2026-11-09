#!/usr/bin/env python3
"""Retarget a segment of LaFAN1 BVH to a 50 Hz Z01 tracking reference using GMR."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from convert_pkl_to_50hz_npz import (
    TARGET_FPS,
    resample_array,
    resample_quaternion,
)
from convert_t4_motion_to_z01 import DEFAULT_XML, Z01_BODY_NAMES, Z01_JOINT_NAMES


def retarget(frames, human_height):
    from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting
    from general_motion_retargeting.params import IK_CONFIG_DICT, ROBOT_XML_DICT

    source = "bvh_lafan1"
    robot = "z01_std_add_head"
    configs = IK_CONFIG_DICT[source]
    config = json.loads(Path(configs["unitree_g1"]).read_text())
    config["robot_root_name"] = "waist_yaw_link"
    for name in ("ik_match_table1", "ik_match_table2"):
        table = config[name]
        table["waist_yaw_link"] = table.pop("pelvis")
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "z01.json"
        path.write_text(json.dumps(config))
        configs[robot] = path
        ROBOT_XML_DICT[robot] = DEFAULT_XML
        try:
            solver = GeneralMotionRetargeting(source, robot, human_height, verbose=False)
        finally:
            del configs[robot]
            del ROBOT_XML_DICT[robot]
        return np.stack([solver.retarget(frame).copy() for frame in frames])


def export_reference(poses, source_fps, output):
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_XML))
    data = mujoco.MjData(model)
    joint_columns = [model.joint(name).qposadr[0] for name in Z01_JOINT_NAMES]
    body_indices = [model.body(name).id for name in Z01_BODY_NAMES]
    foot_geoms = [
        model.geom(f"{side}_foot{index}_collision").id
        for side in ("left", "right")
        for index in range(1, 8)
    ]
    positions = resample_array(poses[:, :3], source_fps, TARGET_FPS)
    positions[:, :2] -= positions[0, :2].copy()
    rotations = resample_quaternion(poses[:, 3:7], source_fps, TARGET_FPS)
    joints = resample_array(poses[:, joint_columns], source_fps, TARGET_FPS)
    joints[:, -2:] = 0.0

    body_positions = []
    body_rotations = []
    support_heights = []
    radii, half_lengths = model.geom_size[foot_geoms, :2].T
    for position, rotation, joint_pos in zip(positions, rotations, joints, strict=True):
        data.qpos[:3] = position
        data.qpos[3:7] = rotation
        data.qpos[joint_columns] = joint_pos
        mujoco.mj_kinematics(model, data)
        body_positions.append(data.xpos[body_indices].copy())
        body_rotations.append(data.xquat[body_indices].copy())
        foot_bottom = (
            data.geom_xpos[foot_geoms, 2]
            - radii
            - half_lengths * np.abs(data.geom_xmat[foot_geoms, 8])
        )
        support_heights.append(foot_bottom.min())

    body_positions = np.stack(body_positions)
    body_rotations = np.stack(body_rotations)
    support_heights = np.asarray(support_heights)
    # Use one ground offset for the clip and remove remaining sole penetration.
    ground = np.percentile(support_heights, 10)
    body_positions[:, :, 2] -= np.minimum(support_heights, ground)[:, None]

    previous = Rotation.from_quat(body_rotations[:-1].reshape(-1, 4), scalar_first=True)
    following = Rotation.from_quat(body_rotations[1:].reshape(-1, 4), scalar_first=True)
    delta = (following * previous.inv()).as_rotvec().reshape(body_positions[1:].shape)
    delta *= TARGET_FPS
    angular_velocity = np.concatenate(
        (delta[:1], (delta[:-1] + delta[1:]) / 2.0, delta[-1:]), axis=0
    )
    np.savez_compressed(
        output,
        fps=np.array([TARGET_FPS], dtype=np.float32),
        joint_pos=joints.astype(np.float32),
        joint_vel=np.gradient(joints, 1.0 / TARGET_FPS, axis=0).astype(np.float32),
        body_pos_w=body_positions.astype(np.float32),
        body_quat_w=body_rotations.astype(np.float32),
        body_lin_vel_w=np.gradient(body_positions, 1.0 / TARGET_FPS, axis=0).astype(np.float32),
        body_ang_vel_w=angular_velocity.astype(np.float32),
    )
    planar_speed = np.linalg.norm(np.diff(body_positions[:, 0, :2], axis=0), axis=-1)
    print(f"{output}: {len(joints)} frames, peak root speed {planar_speed.max() * TARGET_FPS:.2f} m/s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=int, default=0, help="First source frame")
    parser.add_argument("--end", type=int, required=True, help="Exclusive source frame")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    args = parser.parse_args()
    if not 0 <= args.start < args.end - 1 or not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("Choose at least two frames and a positive, finite speed.")
    if args.output.suffix != ".npz":
        parser.error("The output file must end in .npz.")

    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    with args.motion.open() as source:
        frame_time = next(
            float(line.split(":", 1)[1]) for line in source if line.startswith("Frame Time:")
        )
    frames, height = load_bvh_file(str(args.motion))
    if args.end > len(frames):
        parser.error(f"The source has only {len(frames)} frames.")
    poses = retarget(frames[args.start:args.end], height)
    export_reference(poses, args.speed / frame_time, args.output)


if __name__ == "__main__":
    main()
