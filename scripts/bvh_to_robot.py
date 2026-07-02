import argparse
import pathlib
import time
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.lafan1 import load_bvh_file
import mujoco as mj
from rich import print
from tqdm import tqdm
import os
import numpy as np


def parse_name_value_pairs(items, option_name):
    values = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid {option_name} '{item}'. Expected name=value.")
        name, value = item.split("=", 1)
        values[name] = float(value)
    return values


def load_qpos_file(path):
    qpos_path = pathlib.Path(path)
    if qpos_path.suffix == ".npy":
        return np.load(qpos_path)

    text = qpos_path.read_text()
    values = text.replace(",", " ").split()
    return np.array([float(value) for value in values], dtype=float)


def build_initial_qpos(retargeter, args):
    if args.initial_qpos is not None:
        qpos = load_qpos_file(args.initial_qpos)
        if qpos.shape != (retargeter.model.nq,):
            raise ValueError(
                f"--initial_qpos must contain {retargeter.model.nq} values, "
                f"got shape {qpos.shape}"
            )
        return qpos.copy()

    qpos = retargeter.model.qpos0.copy()

    if args.initial_root_pos is not None:
        qpos[:3] = np.array(args.initial_root_pos)

    if args.initial_root_quat is not None:
        root_quat = np.array(args.initial_root_quat, dtype=float)
        quat_norm = np.linalg.norm(root_quat)
        if quat_norm == 0.0:
            raise ValueError("--initial_root_quat must not be a zero quaternion")
        qpos[3:7] = root_quat / quat_norm

    joint_values_deg = parse_name_value_pairs(args.initial_joint, "--initial_joint")
    for joint_name, value_deg in joint_values_deg.items():
        joint_id = mj.mj_name2id(
            retargeter.model,
            mj.mjtObj.mjOBJ_JOINT,
            joint_name,
        )
        if joint_id < 0:
            raise ValueError(f"Joint '{joint_name}' does not exist in the robot model")

        qpos_adr = retargeter.model.jnt_qposadr[joint_id]
        joint_type = retargeter.model.jnt_type[joint_id]
        if qpos_adr < 7:
            raise ValueError(
                f"Joint '{joint_name}' belongs to the floating base; use "
                "--initial_root_pos or --initial_root_quat instead"
            )

        if joint_type == mj.mjtJoint.mjJNT_HINGE:
            qpos[qpos_adr] = np.deg2rad(value_deg)
        else:
            qpos[qpos_adr] = value_deg

    return qpos


if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bvh_file",
        help="BVH motion file to load.",
        required=True,
        type=str,
    )
    
    parser.add_argument(
        "--format",
        choices=["lafan1", "nokov", "mocap58"],
        default="lafan1",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01", "pal_talos", "d20_v2"],
        default="unitree_g1",
    )
    
    
    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--no_viewer",
        action="store_true",
        default=False,
        help="Run retargeting offline without opening the MuJoCo viewer.",
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default="videos/example.mp4",
    )

    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--follow_camera",
        action="store_true",
        default=True,
        help=(
            "Keep camera locked to the robot. By default the camera is left "
            "interactive so MuJoCo mouse controls work during playback."
        ),
    )

    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    
    parser.add_argument(
        "--motion_fps",
        default=30,
        type=int,
    )

    parser.add_argument(
        "--debug_frame",
        default=-1,
        type=int,
        help="Pause the visualization at this frame index for parameter tuning. -1 disables.",
    )
    parser.add_argument(
        "--initial_qpos",
        default=None,
        help=(
            "Path to a full initial qpos vector. Supports .npy or text with "
            "whitespace/comma-separated values. Overrides other initial pose args."
        ),
    )
    parser.add_argument(
        "--initial_root_pos",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Initial floating-base root position for the first IK frame.",
    )
    parser.add_argument(
        "--initial_root_quat",
        type=float,
        nargs=4,
        default=None,
        metavar=("W", "X", "Y", "Z"),
        help="Initial floating-base root quaternion for the first IK frame.",
    )
    parser.add_argument(
        "--initial_joint",
        action="append",
        default=[],
        help=(
            "Initial joint value as joint_name=degrees for hinge joints. "
            "Can be repeated."
        ),
    )

    args = parser.parse_args()
    
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    
    # Load SMPLX trajectory
    lafan1_data_frames, actual_human_height = load_bvh_file(args.bvh_file, format=args.format)
    
    
    # Initialize the retargeting system
    retargeter = GMR(
        src_human=f"bvh_{args.format}",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
    )

    if (
        args.initial_qpos is not None
        or args.initial_root_pos is not None
        or args.initial_root_quat is not None
        or args.initial_joint
    ):
        initial_qpos = build_initial_qpos(retargeter, args)
        retargeter.configuration.update(initial_qpos)
        print("[bold]Applied manual initial IK guess for first frame.[/]")

    motion_fps = args.motion_fps
    
    robot_motion_viewer = None
    if not args.no_viewer:
        robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                                motion_fps=motion_fps,
                                                transparent_robot=1,
                                                record_video=args.record_video,
                                                video_path=args.video_path,
                                                # video_width=2080,
                                                # video_height=1170
                                                )
    
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    print(f"mocap_frame_rate: {motion_fps}")
    
    # Create tqdm progress bar for the total number of frames
    pbar = tqdm(total=len(lafan1_data_frames), desc="Retargeting")
    
    # Start the viewer
    i = 0
    


    while True:
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            fps_label = "processing" if args.no_viewer else "rendering"
            print(f"Actual {fps_label} FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
            
        # Update progress bar
        pbar.update(1)

        # Update task targets.
        smplx_data = lafan1_data_frames[i]

        # retarget
        qpos = retargeter.retarget(smplx_data)
        if args.save_path is not None:
            qpos_list.append(qpos.copy())

        # visualize
        if robot_motion_viewer is not None:
            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retargeter.scaled_human_data,
                rate_limit=args.rate_limit,
                follow_camera=args.follow_camera,
                human_pos_offset=np.array([0.0, 0.0, 0.0])
            )

        # Debug pause: freeze at the specified frame for parameter tuning
        if i == args.debug_frame:
            print(f"\n[DEBUG] Paused at frame {i}. Press Enter to resume.")
            input()
            print(f"[DEBUG] Resuming from frame {i}.")

        if args.loop:
            i = (i + 1) % len(lafan1_data_frames)
        else:
            i += 1
            if i >= len(lafan1_data_frames):
                break
   
    if args.save_path is not None:
        import pickle
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        local_body_pos = None
        body_names = None
        
        motion_data = {
            "fps": motion_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
        }

        # Drop the first 5 frames
        motion_data["root_pos"] = motion_data["root_pos"][5:]
        motion_data["root_rot"] = motion_data["root_rot"][5:]
        motion_data["dof_pos"] = motion_data["dof_pos"][5:]
        if local_body_pos:
            motion_data["local_body_pos"] = motion_data["local_body_pos"][5:]
            

        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    # Close progress bar
    pbar.close()

    if robot_motion_viewer is not None:
        robot_motion_viewer.close()
       
