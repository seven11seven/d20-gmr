"""
Visualize a static mocap58 scaled human skeleton alongside the D20 robot.

Usage:
    python scripts/vis_mocap58_bvh.py --bvh_file data/mocap58/Xsens1/4.bvh
    python scripts/vis_mocap58_bvh.py --bvh_file data/mocap58/Xsens1/4.bvh --pose_source frame --frame_idx 0
"""

import argparse
import pathlib

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R
from rich import print

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.lafan_vendor import utils as lafan_utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh


def build_connections(anim) -> list[tuple[str, str]]:
    connections = []
    for i, bone in enumerate(anim.bones):
        parent_idx = anim.parents[i]
        if parent_idx >= 0:
            connections.append((anim.bones[parent_idx], bone))
    return connections


def build_visible_connections(anim, visible_body_names: set[str]) -> list[tuple[str, str]]:
    connections = []
    for bone_idx, bone in enumerate(anim.bones):
        if bone not in visible_body_names:
            continue

        parent_idx = anim.parents[bone_idx]
        while parent_idx >= 0:
            parent_name = anim.bones[parent_idx]
            if parent_name in visible_body_names:
                connections.append((parent_name, bone))
                break
            parent_idx = anim.parents[parent_idx]

    return connections


def compute_world_pose(
    anim,
    *,
    frame_idx: int = 0,
    use_rest_pose: bool = True,
    center_xy: bool = True,
) -> tuple[dict, np.ndarray]:
    if use_rest_pose:
        local_quats = np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            (1, len(anim.bones), 1),
        )
        local_pos = anim.offsets[np.newaxis, :, :].copy()
    else:
        if frame_idx < 0 or frame_idx >= anim.pos.shape[0]:
            raise ValueError(
                f"frame_idx {frame_idx} is out of range for BVH with "
                f"{anim.pos.shape[0]} frames"
            )
        local_quats = anim.quats[frame_idx:frame_idx + 1]
        local_pos = anim.pos[frame_idx:frame_idx + 1]

    global_quats, global_positions = lafan_utils.quat_fk(
        local_quats,
        local_pos,
        anim.parents,
    )

    # mocap58 BVH coordinates are X-left, Y-up, Z-forward. The project world is
    # X-forward, Y-left, Z-up, so positions become (Z, X, Y).
    rotation_matrix = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

    positions = global_positions[0] @ rotation_matrix.T
    if center_xy:
        positions[:, :2] -= positions[0, :2]

    pose = {}
    for bone_idx, bone in enumerate(anim.bones):
        quat = lafan_utils.quat_mul(rotation_quat, global_quats[0, bone_idx])
        pose[bone] = [positions[bone_idx], quat]

    return pose, positions[0]


def reposition_human_root(human_data: dict, root_name: str, target_pos: np.ndarray) -> dict:
    root_pos = human_data[root_name][0]
    delta = target_pos - root_pos
    return {
        body_name: [pos + delta, quat]
        for body_name, (pos, quat) in human_data.items()
    }


def scale_and_offset_human_data(retargeter, human_data: dict) -> dict:
    """Match GeneralMotionRetargeting.__call__ scale/offset preprocessing."""
    human_data = retargeter.to_numpy(human_data)
    human_data = retargeter.scale_human_data(
        human_data,
        retargeter.human_root_name,
        retargeter.human_scale_table,
    )
    human_data = retargeter.offset_human_data(
        human_data,
        retargeter.pos_offsets1,
        retargeter.rot_offsets1,
    )
    return human_data


def parse_joint_offsets(offset_args: list[str]) -> dict[str, float]:
    offsets = {}
    for item in offset_args:
        if "=" not in item:
            raise ValueError(
                f"Invalid --joint_offset '{item}'. Expected format joint_name=degrees."
            )
        name, value = item.split("=", 1)
        offsets[name] = float(value)
    return offsets


def build_robot_qpos(retargeter, root_height, root_rpy_deg, joint_offsets_deg):
    dof_name_to_qpos_idx = {}
    for i in range(retargeter.model.nv):
        jnt_id = retargeter.model.dof_jntid[i]
        if jnt_id < 0:
            continue
        qpos_adr = retargeter.model.jnt_qposadr[jnt_id]
        if qpos_adr < 7:
            continue
        name = mj.mj_id2name(retargeter.model, mj.mjtObj.mjOBJ_JOINT, jnt_id)
        if name and name not in dof_name_to_qpos_idx:
            dof_name_to_qpos_idx[name] = qpos_adr

    root_rot_quat = R.from_euler("xyz", np.deg2rad(root_rpy_deg)).as_quat(
        scalar_first=True
    )

    robot_qpos = retargeter.model.qpos0.copy()
    robot_qpos[2] = root_height
    robot_qpos[3:7] = root_rot_quat

    for joint_name, offset_deg in joint_offsets_deg.items():
        if joint_name in dof_name_to_qpos_idx:
            robot_qpos[dof_name_to_qpos_idx[joint_name]] += np.deg2rad(offset_deg)
        else:
            print(f"  [yellow]Warning:[/] joint '{joint_name}' not found in robot model")

    return robot_qpos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare a static mocap58 scaled human skeleton with D20 in MuJoCo."
    )
    parser.add_argument(
        "--bvh_file",
        default="data/mocap58/Xsens1/4.bvh",
        help="Path to a mocap58 BVH file.",
    )
    parser.add_argument(
        "--pose_source",
        default="rest",
        choices=["rest", "frame"],
        help="Human pose source. 'rest' uses BVH offsets; 'frame' uses a BVH frame.",
    )
    parser.add_argument(
        "--frame_idx",
        type=int,
        default=0,
        help="BVH frame index used when --pose_source frame.",
    )
    parser.add_argument(
        "--show_body_names",
        action="store_true",
        default=True,
        help="Show joint labels on the coordinate frames.",
    )
    parser.add_argument(
        "--no_show_body_names",
        action="store_false",
        dest="show_body_names",
        help="Hide joint labels.",
    )
    parser.add_argument(
        "--keep_global_position",
        action="store_true",
        default=False,
        help="Do not recenter the first root XY position to the world origin.",
    )
    parser.add_argument(
        "--robot",
        default="d20_v2",
        choices=["d20_v2"],
        help="Robot model to compare against.",
    )
    parser.add_argument(
        "--human_height",
        type=float,
        default=None,
        help="Actual human height in meters. Defaults to the IK config assumption.",
    )
    parser.add_argument(
        "--show_raw_skeleton",
        action="store_true",
        default=False,
        help="Show unscaled mocap58 skeleton instead of GMR-scaled target skeleton.",
    )
    parser.add_argument(
        "--root_height",
        type=float,
        default=1.0,
        help="Root Z position for both human and robot skeletons.",
    )
    parser.add_argument(
        "--human_offset_x",
        type=float,
        default=0.0,
        help="Offset the human skeleton in X from the shared root position.",
    )
    parser.add_argument(
        "--human_offset_y",
        type=float,
        default=0.0,
        help="Offset the human skeleton in Y from the shared root position.",
    )
    parser.add_argument(
        "--align_human_root",
        action="store_true",
        default=True,
        help="Translate the displayed human root to --root_height and human XY offsets.",
    )
    parser.add_argument(
        "--no_align_human_root",
        action="store_false",
        dest="align_human_root",
        help="Show the exact skeleton after GMR scale_human_data() and offset_human_data().",
    )
    parser.add_argument(
        "--robot_root_rpy_deg",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("ROLL", "PITCH", "YAW"),
        help="Robot root RPY in degrees.",
    )
    parser.add_argument(
        "--joint_offset",
        action="append",
        default=[],
        help="Additional robot joint offset as joint_name=degrees. Can be repeated.",
    )
    parser.add_argument(
        "--frame_size",
        type=float,
        default=0.06,
        help="Coordinate frame arrow length.",
    )
    parser.add_argument(
        "--line_width",
        type=float,
        default=0.008,
        help="Skeleton capsule line width.",
    )
    args = parser.parse_args()

    bvh_file = pathlib.Path(args.bvh_file)
    print(f"[bold]Loading mocap58 BVH:[/] {bvh_file}")
    anim = read_bvh(str(bvh_file))
    raw_connections = build_connections(anim)
    raw_human_data, initial_root = compute_world_pose(
        anim,
        frame_idx=args.frame_idx,
        use_rest_pose=args.pose_source == "rest",
        center_xy=not args.keep_global_position,
    )

    print(f"[bold]Initializing GMR for robot:[/] {args.robot}")
    retargeter = GMR(
        src_human="bvh_mocap58",
        tgt_robot=args.robot,
        actual_human_height=args.human_height,
        verbose=False,
    )
    scaled_connections = build_visible_connections(
        anim,
        set(retargeter.human_scale_table.keys()),
    )

    print(f"  BVH frames:   {anim.pos.shape[0]}")
    print(f"  Bones:        {len(anim.bones)}")
    print(f"  Raw links:    {len(raw_connections)}")
    print(f"  Scaled links: {len(scaled_connections)}")
    print(f"  Pose source:  {args.pose_source}{f' frame {args.frame_idx}' if args.pose_source == 'frame' else ''}")
    print(f"  Initial root: X={initial_root[0]:.3f} Y={initial_root[1]:.3f} Z={initial_root[2]:.3f}")

    human_data_scaled_offset = scale_and_offset_human_data(
        retargeter,
        raw_human_data.copy(),
    )

    human_root_target = np.array([
        args.human_offset_x,
        args.human_offset_y,
        args.root_height,
    ])

    if args.align_human_root:
        human_data_display = reposition_human_root(
            human_data_scaled_offset,
            retargeter.human_root_name,
            human_root_target,
        )
        raw_human_data = reposition_human_root(
            raw_human_data,
            retargeter.human_root_name,
            human_root_target,
        )
    else:
        human_data_display = human_data_scaled_offset

    display_human_data = raw_human_data if args.show_raw_skeleton else human_data_display
    display_connections = raw_connections if args.show_raw_skeleton else scaled_connections

    default_joint_offsets = {
        "left_shoulder_roll_joint": 90.0,
        "right_shoulder_roll_joint": -90.0,
        "left_elbow_pitch_joint": 90.0,
        "right_elbow_pitch_joint": 90.0,
    }
    default_joint_offsets.update(parse_joint_offsets(args.joint_offset))

    robot_qpos = build_robot_qpos(
        retargeter,
        args.root_height,
        args.robot_root_rpy_deg,
        default_joint_offsets,
    )
    robot_root = [
        float(robot_qpos[0]),
        float(robot_qpos[1]),
        float(robot_qpos[2]),
    ]
    step_size = [0.01]

    print(f"  Robot root RPY: roll={args.robot_root_rpy_deg[0]:.1f} "
          f"pitch={args.robot_root_rpy_deg[1]:.1f} yaw={args.robot_root_rpy_deg[2]:.1f}")
    if default_joint_offsets:
        print("  Joint offsets applied:")
        for name, deg in default_joint_offsets.items():
            print(f"    {name}: {deg:+.1f}°")

    KEY_R = 82
    KEY_1 = 49
    KEY_2 = 50
    KEY_3 = 51
    KEY_UP = 265
    KEY_DOWN = 264
    KEY_LEFT = 263
    KEY_RIGHT = 262
    KEY_KP_2 = 322
    KEY_KP_8 = 328

    def make_keyboard_callback(robot_root, step_size, root_height):
        def on_key(keycode: int) -> None:
            step = step_size[0]
            if keycode == KEY_UP:
                robot_root[0] += step
            elif keycode == KEY_DOWN:
                robot_root[0] -= step
            elif keycode == KEY_LEFT:
                robot_root[1] -= step
            elif keycode == KEY_RIGHT:
                robot_root[1] += step
            elif keycode == KEY_KP_2:
                robot_root[2] -= step
            elif keycode == KEY_KP_8:
                robot_root[2] += step
            elif keycode == KEY_R:
                robot_root[0] = 0.0
                robot_root[1] = 0.0
                robot_root[2] = root_height
            elif keycode == KEY_1:
                step_size[0] = 0.01
            elif keycode == KEY_2:
                step_size[0] = 0.05
            elif keycode == KEY_3:
                step_size[0] = 0.10
            else:
                return

            print(
                f"  Robot root: X={robot_root[0]:.3f}  Y={robot_root[1]:.3f}  Z={robot_root[2]:.3f}"
                f"  |  step={step_size[0]:.2f}m"
            )

        return on_key

    viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=30,
        transparent_robot=1,
        keyboard_callback=make_keyboard_callback(robot_root, step_size, args.root_height),
    )
    viewer.viewer.cam.lookat = [0.0, 0.0, args.root_height]
    viewer.viewer.cam.distance = viewer.viewer_cam_distance
    viewer.viewer.cam.elevation = -15

    print("\n[bold green]Launching static mocap58 comparison viewer[/]")
    print(f"  Robot:          {args.robot}  (transparent)")
    print(f"  Human:          {bvh_file}")
    print(f"  Skeleton:       {'raw mocap58' if args.show_raw_skeleton else 'GMR scale_human_data + offset_human_data'}")
    print(f"  Human align:    {'enabled' if args.align_human_root else 'disabled'}")
    print(f"  Root height:    {args.root_height} m")
    print(f"  Human offset:   X={args.human_offset_x}, Y={args.human_offset_y}")
    print("")
    print("  [bold]Robot controls (focus the MuJoCo window):[/]")
    print("    [cyan]↑/↓[/]           — move robot  forward / backward  (X axis)")
    print("    [cyan]←/→[/]           — move robot  left / right       (Y axis)")
    print("    [cyan]Numpad 2/8[/]    — move robot  down / up          (Z axis)")
    print(f"    [cyan]R[/]             — reset robot to (0, 0, {args.root_height})")
    print("    [cyan]1/2/3[/]         — step size: 0.01 / 0.05 / 0.10 m")
    print("")
    print("  [dim]Left-drag to rotate | Right-drag to pan | Scroll to zoom[/]")
    print("  [dim]Close the viewer window to exit.[/]")

    try:
        while viewer.viewer.is_running():
            viewer.step(
                root_pos=np.array(robot_root),
                root_rot=robot_qpos[3:7],
                dof_pos=robot_qpos[7:],
                human_motion_data=display_human_data,
                show_human_body_name=args.show_body_names,
                human_point_scale=args.frame_size,
                human_pos_offset=np.zeros(3),
                human_body_connections=display_connections,
                skeleton_line_width=args.line_width,
                follow_camera=False,
                rate_limit=True,
            )
    except KeyboardInterrupt:
        print("\n[bold yellow]Interrupted by user.[/]")
    finally:
        viewer.close()
        print("[bold]Viewer closed.[/]")
