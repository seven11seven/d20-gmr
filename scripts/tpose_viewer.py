"""
T-Pose Viewer: Visualize the scaled human skeleton alongside the robot in T-pose.

1. Loads a BVH file to extract the skeleton definition (bone names, offsets, hierarchy)
2. Computes a human pose from the BVH hierarchy, using either a real BVH frame
   or the raw zero-rotation rest pose
3. Applies the scaling/offset pipeline from the IK config (same as retargeting pipeline)
4. Displays both the scaled human skeleton (as coordinate-frame arrows) and
   the robot (in transparent mode) in a static MuJoCo viewer.

Usage:
    python scripts/tpose_viewer.py --bvh_file data/lafan/dance1_subject2.bvh --robot d20_v2
    python scripts/tpose_viewer.py --bvh_file data/lafan/dance1_subject2.bvh --robot d20_v2 --pose_source rest
"""

import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco as mj
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh
from general_motion_retargeting.utils.lafan_vendor import utils as lafan_utils
from rich import print


def create_human_pose_from_anim(
    anim,
    *,
    frame_idx: int = 0,
    use_rest_pose: bool = False,
    bvh_format: str = "lafan1",
) -> tuple[dict, list]:
    """
    Create a human pose from a parsed BVH Anim object.

    LaFan's zero-rotation OFFSET pose is not an anatomical standing T-pose:
    many limbs extend along the BVH X axis. For visual debugging, a real BVH
    frame is usually the least surprising source because the joint names,
    rotations, and FK positions all come from the same pose.

    Returns:
        human_data: dict mapping bone names → [3D position, quaternion (scalar-first)]
        connections: list of (parent_name, child_name) tuples for skeleton links
    """
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
        local_quats, local_pos, anim.parents
    )

    # Apply the same coordinate transform used by load_bvh_file():
    #   BVH coords (X-forward, Y-up, Z-left) → world coords (X-forward, Y-left, Z-up)
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

    result = {}
    for i, bone in enumerate(anim.bones):
        orientation = lafan_utils.quat_mul(rotation_quat, global_quats[0, i])
        position = global_positions[0, i] @ rotation_matrix.T / 100.0  # cm → m
        result[bone] = [position, orientation]

    if bvh_format == "lafan1":
        left_toe_name = "LeftToe"
        right_toe_name = "RightToe"
    elif bvh_format == "nokov":
        left_toe_name = "LeftToeBase"
        right_toe_name = "RightToeBase"
    else:
        raise ValueError(f"Invalid format: {bvh_format}")

    # Add synthetic foot task bones required by the IK config. Their positions
    # are foot/ankle positions, while their orientations follow the toe.
    result["LeftFootMod"] = [result["LeftFoot"][0].copy(), result[left_toe_name][1]]
    result["RightFootMod"] = [result["RightFoot"][0].copy(), result[right_toe_name][1]]

    # Build parent→child connections from the BVH hierarchy
    connections = []
    for i, bone in enumerate(anim.bones):
        parent_idx = anim.parents[i]
        if parent_idx >= 0:
            connections.append((anim.bones[parent_idx], bone))

    return result, connections

def reposition_human_root(human_data: dict, root_name: str, target_pos: np.ndarray) -> dict:
    """
    Reposition the human skeleton so the root bone sits at `target_pos`.
    All other bones are translated by the same delta.
    """
    root_pos = human_data[root_name][0]
    delta = target_pos - root_pos

    repositioned = {}
    for body_name, (pos, quat) in human_data.items():
        repositioned[body_name] = [pos + delta, quat]
    return repositioned

def scale_all_bones(
    human_data: dict,
    human_scale_table: dict,
    human_root_name: str,
    anim_parents: list,
    anim_bones: list,
) -> dict:
    """
    Scale human data exactly like GeneralMotionRetargeting.scale_human_data().

    Only bones listed in the IK config's human_scale_table are preserved.
    Positions are scaled in the root-local frame and transformed back to global.
    anim_parents and anim_bones are accepted for API compatibility with older
    viewer code, but the retargeting scale path does not use hierarchy.
    """
    _ = anim_parents, anim_bones

    human_data_local = {}
    root_pos, root_quat = human_data[human_root_name]

    scaled_root_pos = human_scale_table[human_root_name] * root_pos

    for body_name in human_data.keys():
        if body_name not in human_scale_table:
            continue
        if body_name == human_root_name:
            continue
        human_data_local[body_name] = (
            human_data[body_name][0] - root_pos
        ) * human_scale_table[body_name]

    result = {human_root_name: (scaled_root_pos, root_quat)}
    for body_name in human_data_local.keys():
        result[body_name] = (
            human_data_local[body_name] + scaled_root_pos,
            human_data[body_name][1],
        )

    return result


def filter_skeleton_connections(connections: list, human_data: dict) -> list:
    """Keep only visible links whose endpoints survived retargeting-scale filtering."""
    visible_connections = [
        (parent_name, child_name)
        for parent_name, child_name in connections
        if parent_name in human_data and child_name in human_data
    ]

    # The retargeting scale path intentionally drops BVH Foot joints because the
    # IK config uses FootMod task bones instead. Add display links that preserve
    # the visible lower-leg and foot-arch structure after that filtering.
    foot_links = [
        ("LeftLeg", "LeftFootMod"),
        ("LeftFootMod", "LeftToe"),
        ("RightLeg", "RightFootMod"),
        ("RightFootMod", "RightToe"),
    ]
    for parent_name, child_name in foot_links:
        link = (parent_name, child_name)
        if (
            parent_name in human_data
            and child_name in human_data
            and link not in visible_connections
        ):
            visible_connections.append(link)

    return visible_connections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize scaled human skeleton and robot in T-pose"
    )
    parser.add_argument(
        "--bvh_file",
        required=True,
        help="Path to BVH file (used for skeleton definition / bone names).",
    )
    parser.add_argument(
        "--robot",
        default="d20_v2",
        help="Robot name (must have a matching IK config for the BVH format).",
    )
    parser.add_argument(
        "--format",
        default="lafan1",
        choices=["lafan1", "nokov"],
        help="BVH format (determines which IK config set to use).",
    )
    parser.add_argument(
        "--human_height",
        type=float,
        default=1.75,
        help="Actual human height in meters (for scaling ratio vs config assumption).",
    )
    parser.add_argument(
        "--root_height",
        type=float,
        default=1.8,
        help="Root (Hips / base_link) Z position for both human and robot skeletons.",
    )
    parser.add_argument(
        "--pose_source",
        default="frame",
        choices=["frame", "rest"],
        help=(
            "Human skeleton pose source. 'frame' uses a real BVH frame; "
            "'rest' uses zero rotations plus BVH OFFSETs."
        ),
    )
    parser.add_argument(
        "--frame_idx",
        type=int,
        default=0,
        help="BVH frame index used when --pose_source frame.",
    )
    parser.add_argument(
        "--human_offset_x",
        type=float,
        default=0.0,
        help="Offset the human skeleton in X (forward) from the shared root position.",
    )
    parser.add_argument(
        "--human_offset_y",
        type=float,
        default=0.0,
        help="Offset the human skeleton in Y (left) from the shared root position.",
    )
    parser.add_argument(
        "--show_body_names",
        action="store_true",
        default=True,
        help="Show human body part name labels on the coordinate frames.",
    )
    parser.add_argument(
        "--no_show_body_names",
        action="store_false",
        dest="show_body_names",
        help="Hide human body part name labels.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Read BVH and build T-pose + skeleton connections
    # ------------------------------------------------------------------
    print(f"[bold]Loading BVH skeleton from:[/] {args.bvh_file}")
    anim = read_bvh(args.bvh_file)
    human_data_full, skeleton_connections = create_human_pose_from_anim(
        anim,
        frame_idx=args.frame_idx,
        use_rest_pose=args.pose_source == "rest",
        bvh_format=args.format,
    )
    print(f"  Extracted {len(human_data_full)} body parts, {len(skeleton_connections)} links")
    if args.pose_source == "frame":
        print(f"  Human pose source: BVH frame {args.frame_idx}")
    else:
        print("  Human pose source: BVH rest pose (zero joint rotations)")

    # ------------------------------------------------------------------
    # 2. Initialize GMR to load robot model and IK config
    # ------------------------------------------------------------------
    print(f"[bold]Initializing GMR for robot:[/] {args.robot}")
    retargeter = GMR(
        src_human=f"bvh_{args.format}",
        tgt_robot=args.robot,
        actual_human_height=args.human_height,
        verbose=False,
    )

    # ------------------------------------------------------------------
    # Robot default-pose configuration
    #   - root_rpy_deg:  [roll, pitch, yaw] in degrees for the root link
    #   - joint_offsets_deg:  dict of {joint_name: offset_degrees}
    #     joint names are the MuJoCo DoF names (see printed list at init)
    # ------------------------------------------------------------------
    robot_root_rpy_deg = [0.0, 0.0, -90.0]  # roll, pitch, yaw in degrees

    robot_joint_offsets_deg = {
        # Examples (uncomment to activate):
        # "left_shoulder_pitch_joint":  30.0,   # raise left arm  30°
        # "right_shoulder_pitch_joint": 30.0,   # raise right arm 30°
        # "left_shoulder_yaw_joint":   -60.0,   # swing left arm  out
        
        "left_shoulder_roll_joint":   90.0,   # swing left arm  out
        "right_shoulder_roll_joint": -90.0,   # swing left arm  out
        
        "left_elbow_pitch_joint":    90.0,   # swing left arm  out
        "right_elbow_pitch_joint":   90.0,   # swing left arm  out
        
        # "right_shoulder_yaw_joint":   60.0,   # swing right arm out
        # "waist_yaw_joint":            15.0,   # twist torso
        # "left_hip_pitch_joint":      -20.0,   # lean left leg forward
        # "right_hip_pitch_joint":     -20.0,   # lean right leg forward
        # "left_knee_pitch_joint":      40.0,   # bend left knee
        # "right_knee_pitch_joint":     40.0,   # bend right knee
    }

    # Build DoF name → qpos index lookup (only hinge/slide joints, skip floating base)
    dof_name_to_qpos_idx = {}
    for i in range(retargeter.model.nv):
        jnt_id = retargeter.model.dof_jntid[i]
        if jnt_id < 0:
            continue  # floating-base velocity DoF (no corresponding joint)
        qpos_adr = retargeter.model.jnt_qposadr[jnt_id]
        if qpos_adr < 7:
            continue  # floating-base joint itself (occupies qpos 0..6)
        name = mj.mj_id2name(retargeter.model, mj.mjtObj.mjOBJ_JOINT, jnt_id)
        if name and name not in dof_name_to_qpos_idx:
            dof_name_to_qpos_idx[name] = qpos_adr

    # Convert root RPY → quaternion (scalar-first)
    root_rot_quat = R.from_euler("xyz", np.deg2rad(robot_root_rpy_deg)).as_quat(scalar_first=True)

    # Build robot qpos: start from default, apply root height, root rotation, and joint offsets
    robot_qpos = retargeter.model.qpos0.copy()
    robot_qpos[2] = args.root_height
    robot_qpos[3:7] = root_rot_quat

    for joint_name, offset_deg in robot_joint_offsets_deg.items():
        if joint_name in dof_name_to_qpos_idx:
            idx = dof_name_to_qpos_idx[joint_name]
            robot_qpos[idx] += np.deg2rad(offset_deg)
        else:
            print(f"  [yellow]Warning:[/] joint '{joint_name}' not found in robot model")

    print(f"  Robot root RPY: roll={robot_root_rpy_deg[0]:.1f}°  "
          f"pitch={robot_root_rpy_deg[1]:.1f}°  yaw={robot_root_rpy_deg[2]:.1f}°")
    if robot_joint_offsets_deg:
        print(f"  Joint offsets applied:")
        for name, deg in robot_joint_offsets_deg.items():
            print(f"    {name}: {deg:+.1f}°")
    # ------------------------------------------------------------------
    human_data_full = retargeter.to_numpy(human_data_full)
    human_data_full = scale_all_bones(
        human_data_full,
        retargeter.human_scale_table,
        retargeter.human_root_name,
        anim.parents,
        anim.bones,
    )
    skeleton_connections = filter_skeleton_connections(
        skeleton_connections,
        human_data_full,
    )
    print(f"  Visible scaled body parts: {len(human_data_full)}")
    print(f"  Visible skeleton links after IK scale filtering: {len(skeleton_connections)}")

    # Apply IK pos/rot offsets only to config-mapped bones
    config_bone_names = set(retargeter.human_scale_table.keys())
    human_data_config = {
        k: v for k, v in human_data_full.items()
        if k in config_bone_names
    }
    human_data_config = retargeter.offset_human_data(
        human_data_config,
        retargeter.pos_offsets1,
        retargeter.rot_offsets1,
    )
    # Write back offset positions to full data
    for bone, (pos, quat) in human_data_config.items():
        human_data_full[bone] = [pos, quat]

    # Position human root (Hips) at the shared root position + user offset
    human_root_target = np.array([
        args.human_offset_x,
        args.human_offset_y,
        args.root_height,
    ])
    human_data_full = reposition_human_root(
        human_data_full, retargeter.human_root_name, human_root_target
    )

    # ------------------------------------------------------------------
    # 4. Robot: initialize mutable state for interactive keyboard control
    # ------------------------------------------------------------------
    # robot_qpos was already built in the config block above with
    # root height, root RPY, and per-joint offsets applied.

    # Mutable robot root position — modified by keyboard callbacks
    robot_root = [
        float(robot_qpos[0]),
        float(robot_qpos[1]),
        float(robot_qpos[2]),
    ]
    step_size = [0.01]  # mutable, toggled by number keys

    print(f"  Robot DOF count: {retargeter.model.nv}")
    print(f"  Robot root pos: X={robot_root[0]:.3f} Y={robot_root[1]:.3f} Z={robot_root[2]:.3f}")

    # ------------------------------------------------------------------
    # 5. Keyboard callback for interactive robot positioning
    # ------------------------------------------------------------------
    # GLFW key codes
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

            # ---- X axis (forward / backward) ----
            if keycode == KEY_UP:
                robot_root[0] += step
            elif keycode == KEY_DOWN:
                robot_root[0] -= step

            # ---- Y axis (left / right) ----
            elif keycode == KEY_LEFT:
                robot_root[1] -= step
            elif keycode == KEY_RIGHT:
                robot_root[1] += step

            # ---- Z axis (down / up) ----
            elif keycode == KEY_KP_2:
                robot_root[2] -= step
            elif keycode == KEY_KP_8:
                robot_root[2] += step

            # ---- Reset position ----
            elif keycode == KEY_R:
                robot_root[0] = 0.0
                robot_root[1] = 0.0
                robot_root[2] = root_height

            # ---- Step size presets ----
            elif keycode == KEY_1:
                step_size[0] = 0.01
            elif keycode == KEY_2:
                step_size[0] = 0.05
            elif keycode == KEY_3:
                step_size[0] = 0.10

            else:
                return  # no change, don't print

            print(
                f"  Robot root: X={robot_root[0]:.3f}  Y={robot_root[1]:.3f}  Z={robot_root[2]:.3f}"
                f"  |  step={step_size[0]:.2f}m"
            )

        return on_key

    key_callback = make_keyboard_callback(robot_root, step_size, args.root_height)

    # ------------------------------------------------------------------
    # 6. Launch MuJoCo viewer  (robot in transparent mode)
    # ------------------------------------------------------------------
    print(f"\n[bold green]Launching T-Pose Viewer[/]")
    print(f"  Robot:          {args.robot}  (transparent)")
    print(f"  Human:          {args.bvh_file}  (coordinate frames)")
    print(f"  Pose source:    {args.pose_source}"
          f"{f' frame {args.frame_idx}' if args.pose_source == 'frame' else ''}")
    print(f"  Root height:    {args.root_height} m  (shared)")
    print(f"  Human offset:   X={args.human_offset_x}, Y={args.human_offset_y}")
    print(f"")
    print(f"  [bold]Robot controls (focus the MuJoCo window):[/]")
    print(f"    [cyan]↑/↓[/]           — move robot  forward / backward  (X axis)")
    print(f"    [cyan]←/→[/]           — move robot  left / right       (Y axis)")
    print(f"    [cyan]Numpad 2/8[/]    — move robot  down / up          (Z axis)")
    print(f"    [cyan]R[/]             — reset robot to (0, 0, {args.root_height})")
    print(f"    [cyan]1/2/3[/]         — step size: 0.01 / 0.05 / 0.10 m")
    print(f"")
    print(f"  [dim]Left-drag to rotate | Right-drag to pan | Scroll to zoom[/]")
    print(f"  [dim]Close the viewer window to exit.[/]")

    viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=30,
        transparent_robot=1,
        keyboard_callback=key_callback,
    )

    # Set initial camera position once — don't override it on each frame
    viewer.viewer.cam.lookat = [0, 0, args.root_height]
    viewer.viewer.cam.distance = viewer.viewer_cam_distance
    viewer.viewer.cam.elevation = -15

    try:
        while viewer.viewer.is_running():
            viewer.step(
                root_pos=np.array(robot_root),
                root_rot=robot_qpos[3:7],
                dof_pos=robot_qpos[7:],
                human_motion_data=human_data_full,
                show_human_body_name=args.show_body_names,
                human_point_scale=0.08,
                human_pos_offset=np.zeros(3),  # already baked into human_data
                human_body_connections=skeleton_connections,
                skeleton_line_width=0.005,
                follow_camera=False,  # let the user control the camera interactively
                rate_limit=True,
            )
    except KeyboardInterrupt:
        print("\n[bold yellow]Interrupted by user.[/]")
    finally:
        viewer.close()
        print("[bold]Viewer closed.[/]")
