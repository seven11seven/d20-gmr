"""
T-Pose Viewer: Visualize the scaled human skeleton alongside the robot in T-pose.

1. Loads a BVH file to extract the skeleton definition (bone names, offsets, hierarchy)
2. Computes the T-pose (rest pose) using identity quaternions and bone offsets
3. Applies the scaling/offset pipeline from the IK config (same as retargeting pipeline)
4. Displays both the scaled human skeleton (as coordinate-frame arrows) and
   the robot (in transparent mode) in a static MuJoCo viewer.

Usage:
    python scripts/tpose_viewer.py --bvh_file data/lafan/dance1_subject2.bvh --robot d20_v2
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


def create_tpose_from_anim(anim) -> tuple[dict, list]:
    """
    Create a T-pose (rest pose) from a parsed BVH Anim object.

    Returns:
        human_data: dict mapping bone names → [3D position, quaternion (scalar-first)]
        connections: list of (parent_name, child_name) tuples for skeleton links
    """
    # Identity quaternions = no joint rotation → rest pose
    identity_quats = np.tile(
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        (1, len(anim.bones), 1),
    )

    # Use bone offsets as local positions for forward kinematics.
    local_pos = anim.offsets[np.newaxis, :, :].copy()
    global_quats, global_positions = lafan_utils.quat_fk(
        identity_quats, local_pos, anim.parents
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

    # Add synthetic foot bones required by the IK config
    result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftToe"][1]]
    result["RightFootMod"] = [result["RightFoot"][0], result["RightToe"][1]]

    # Build parent→child connections from the BVH hierarchy
    connections = []
    for i, bone in enumerate(anim.bones):
        parent_idx = anim.parents[i]
        if parent_idx >= 0:
            connections.append((anim.bones[parent_idx], bone))

    # Replace foot chain endings: LeftLeg→LeftFootMod (skip LeftToe/RightToe)
    connections = [
        (p, c) for (p, c) in connections
        if c not in ("LeftToe", "RightToe")
    ]
    connections = [
        (p, "LeftFootMod") if c == "LeftFoot" else (p, c)
        for (p, c) in connections
    ]
    connections = [
        (p, "RightFootMod") if c == "RightFoot" else (p, c)
        for (p, c) in connections
    ]

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
    Scale all bones including intermediate ones not in the scale table.

    Bones in the scale table use their explicit scale factor.
    Intermediate bones inherit the scale factor from their nearest
    scaled ancestor in the BVH hierarchy.
    """
    # Build parent index lookup
    bone_to_idx = {b: i for i, b in enumerate(anim_bones)}

    # Synthetic bones inherit scale from their source
    _synthetic_sources = {"LeftFootMod": "LeftFoot", "RightFootMod": "RightFoot"}

    # Determine effective scale for every bone via ancestor inheritance
    effective_scale = {}
    for bone in anim_bones:
        if bone in human_scale_table:
            effective_scale[bone] = human_scale_table[bone]
        else:
            # Walk up the hierarchy to find nearest scaled ancestor
            current = bone
            while current not in human_scale_table:
                idx = bone_to_idx.get(current)
                if idx is None or anim_parents[idx] < 0:
                    effective_scale[bone] = 1.0
                    break
                current = anim_bones[anim_parents[idx]]
            else:
                effective_scale[bone] = human_scale_table[current]

    # Handle synthetic bones not in the BVH hierarchy
    for bone in human_data:
        if bone not in effective_scale:
            eff = 1.0
            if bone in _synthetic_sources:
                src = _synthetic_sources[bone]
                eff = effective_scale.get(src, 1.0)
            effective_scale[bone] = eff

    # Scale all bones relative to root
    root_pos, root_quat = human_data[human_root_name]
    scaled_root_pos = effective_scale[human_root_name] * root_pos

    result = {human_root_name: (scaled_root_pos, root_quat)}
    for bone in human_data:
        if bone == human_root_name:
            continue
        pos, quat = human_data[bone]
        offset = pos - root_pos
        scaled_pos = scaled_root_pos + offset * effective_scale[bone]
        result[bone] = [scaled_pos, quat]

    return result


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
    human_data_full, skeleton_connections = create_tpose_from_anim(anim)
    print(f"  Extracted {len(human_data_full)} body parts, {len(skeleton_connections)} links")

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
    robot_root_rpy_deg = [0.0, -90.0, 0.0]  # roll, pitch, yaw in degrees

    robot_joint_offsets_deg = {
        # Examples (uncomment to activate):
        # "left_shoulder_pitch_joint":  30.0,   # raise left arm  30°
        # "right_shoulder_pitch_joint": 30.0,   # raise right arm 30°
        # "left_shoulder_yaw_joint":   -60.0,   # swing left arm  out
        
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
    step_size = [0.05]  # mutable, toggled by number keys

    print(f"  Robot DOF count: {retargeter.model.nv}")
    print(f"  Robot root pos: X={robot_root[0]:.3f} Y={robot_root[1]:.3f} Z={robot_root[2]:.3f}")

    # ------------------------------------------------------------------
    # 5. Keyboard callback for interactive robot positioning
    # ------------------------------------------------------------------
    # GLFW key codes
    KEY_W = 87
    KEY_A = 65
    KEY_S = 83
    KEY_D = 68
    KEY_Q = 81
    KEY_E = 69
    KEY_R = 82
    KEY_1 = 49
    KEY_2 = 50
    KEY_3 = 51
    KEY_UP = 265
    KEY_DOWN = 266
    KEY_LEFT = 263
    KEY_RIGHT = 262

    def make_keyboard_callback(robot_root, step_size, root_height):
        def on_key(keycode: int) -> None:
            step = step_size[0]

            # ---- X axis (forward / backward) ----
            if keycode == KEY_W or keycode == KEY_UP:
                robot_root[0] += step
            elif keycode == KEY_S or keycode == KEY_DOWN:
                robot_root[0] -= step

            # ---- Y axis (left / right) ----
            elif keycode == KEY_A or keycode == KEY_LEFT:
                robot_root[1] -= step
            elif keycode == KEY_D or keycode == KEY_RIGHT:
                robot_root[1] += step

            # ---- Z axis (down / up) ----
            elif keycode == KEY_Q:
                robot_root[2] -= step
            elif keycode == KEY_E:
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
    print(f"  Root height:    {args.root_height} m  (shared)")
    print(f"  Human offset:   X={args.human_offset_x}, Y={args.human_offset_y}")
    print(f"")
    print(f"  [bold]Robot controls (focus the MuJoCo window):[/]")
    print(f"    [cyan]W/S[/] or [cyan]↑/↓[/]   — move robot  forward / backward  (X axis)")
    print(f"    [cyan]A/D[/] or [cyan]←/→[/]   — move robot  left / right       (Y axis)")
    print(f"    [cyan]Q/E[/]           — move robot  down / up          (Z axis)")
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
