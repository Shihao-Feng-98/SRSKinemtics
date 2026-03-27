from IPython import embed
import numpy as np
from scipy.spatial.transform import Rotation as R
import pathlib
from loop_rate_limiters import RateLimiter
from trimesh.scene import Scene
import viser
from viser.extras import ViserUrdf

from src.interval import Interval
from src.srs_kinematics import SRSKinematics, KineStatus, iiwa14


def rotation_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    xyzw = R.from_matrix(rotation).as_quat()
    return np.roll(xyzw, 1)

def wxyz_to_rotation(wxyz) -> np.ndarray:
    xyzw = np.roll(wxyz, -1)
    return R.from_quat(xyzw).as_matrix()

def pos_wxyz_to_homogeneous(pos: np.ndarray,
                            wxyz: np.ndarray = [1, 0, 0, 0]) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = pos
    T[:3, :3] = wxyz_to_rotation(wxyz)
    return T

def viser_name_from_frame(
    scene: Scene,
    frame_name: str,
    root_node_name: str = "/",
) -> str:
    """Given the (unique) name of a frame in our URDF's kinematic tree, return a
    scene node name for viser.

    For a robot manipulator with four frames, that looks like:


            ((shoulder)) == ((elbow))
               / /             |X|
              / /           ((wrist))
         ____/ /____           |X|
        [           ]       [=======]
        [ base_link ]        []   []
        [___________]


    this would map a name like "elbow" to "base_link/shoulder/elbow".
    """
    assert root_node_name.startswith("/")
    assert len(root_node_name) == 1 or not root_node_name.endswith("/")

    frames = []
    while frame_name != scene.graph.base_frame:
        frames.append(frame_name)
        frame_name = scene.graph.transforms.parents[frame_name]
    if root_node_name != "/":
        frames.append(root_node_name)
    return "/".join(frames[::-1])



if __name__ == "__main__":
    vis = viser.ViserServer()
    vis.scene.world_axes.visible = True
    h_grid = vis.scene.add_grid(name="/ground",
                                        width=10,
                                        height=10)
    # left
    frame_base_left = "/robot_base_left"
    h_base_left = vis.scene.add_frame(name=frame_base_left, 
                                    show_axes=True,
                                    axes_length=0.2,
                                    axes_radius=0.005,
                                    position=(0,0,1), 
                                    wxyz=rotation_to_wxyz(np.array([[0,-1,0],[0,0,1],[-1,0,0]])))
    h_robot_left = ViserUrdf(vis,
                            urdf_or_path=pathlib.Path("./assets/iiwa_description/iiwa14_spheres_dense_collision.urdf"),
                            root_node_name=frame_base_left,
                            load_meshes=True,
                            load_collision_meshes=False)
    h_robot_left.update_cfg(np.deg2rad([-45, 45, 0, 90, 0, 45, 0]))
    frame_ee_left = f'{frame_base_left}/visual/{viser_name_from_frame(h_robot_left._urdf.scene, "iiwa_link_ee_kuka")}/frame'
    h_ee_left = vis.scene.add_frame(frame_ee_left,
                                    show_axes=True,
                                    axes_length=0.1,
                                    axes_radius=0.005,
                                    position=(0, 0, 0),
                                    wxyz=(1, 0, 0, 0))
    left_kine = SRSKinematics(iiwa14())
    left_kine.set_user_frame(pos_wxyz_to_homogeneous(h_base_left.position,
                                                     h_base_left.wxyz))

    # right
    frame_base_right = "/robot_base_right"
    h_base_right = vis.scene.add_frame(name=frame_base_right, 
                        show_axes=True,
                        axes_length=0.2,
                        axes_radius=0.005,
                        position=(0,0,1), 
                        wxyz=rotation_to_wxyz(np.array([[0,1,0],[0,0,-1],[-1,0,0]])))
    h_robot_right = ViserUrdf(vis,
                            urdf_or_path=pathlib.Path("./assets/iiwa_description/iiwa14_spheres_dense_collision.urdf"),
                            root_node_name=frame_base_right,
                            load_meshes=True,
                            load_collision_meshes=False)
    h_robot_right.update_cfg(np.deg2rad([45, 45, 0, 90, 0, 45, 0]))
    frame_ee_right = f'{frame_base_right}/visual/{viser_name_from_frame(h_robot_right._urdf.scene, "iiwa_link_ee_kuka")}/frame'
    h_ee_right = vis.scene.add_frame(frame_ee_right,
                                    show_axes=True,
                                    axes_length=0.1,
                                    axes_radius=0.005,
                                    position=(0, 0, 0),
                                    wxyz=(1, 0, 0, 0))
    right_kine = SRSKinematics(iiwa14())
    right_kine.set_user_frame(pos_wxyz_to_homogeneous(h_base_right.position, 
                                                      h_base_right.wxyz))

    # 身体
    sphere_positions = [(0,0,0.2),(0,0,0.4),(0,0,0.6),(0,0,0.8)]
    sphere_colors = [[150,150,150]]*len(sphere_positions)
    h_spheres = []
    for i, (pos, color) in enumerate(zip(sphere_positions, sphere_colors)):
        h_sphere = vis.scene.add_icosphere(name=f"/sphere_{i}", radius=0.25, position=pos, color=color)
        h_spheres.append(h_sphere)

    # 状态显示
    # left
    vis.gui.configure_theme(control_width="large")
    h_left_qpos_text = vis.gui.add_text(label="left_qpos(rad)", initial_value="[]")
    h_left_arm_angle_text = vis.gui.add_text(label="left_arm_angle(rad)", initial_value="[]")
    h_left_ee_pose_text = vis.gui.add_text(label="left_ee_pose", initial_value="[]")

    cur_left_pose = left_kine.get_fk(h_robot_left._urdf.cfg.copy())
    h_ee_goal_left = vis.scene.add_transform_controls(name="/ee_target_left",
                                                                scale=0.2,
                                                                position=cur_left_pose[:3, 3],
                                                                wxyz=rotation_to_wxyz(cur_left_pose[:3, :3]))
    # right
    h_right_qpos_text = vis.gui.add_text(label="right_qpos(rad)", initial_value="[]")
    h_right_arm_angle_text = vis.gui.add_text(label="right_arm_angle(rad)", initial_value="[]")
    h_right_ee_pose_text = vis.gui.add_text(label="right_ee_pose", initial_value="[]")

    cur_right_pose = right_kine.get_fk(h_robot_right._urdf.cfg.copy())
    h_ee_goal_right = vis.scene.add_transform_controls(name="/ee_target_right",
                                                                scale=0.2,
                                                                position=cur_right_pose[:3, 3],
                                                                wxyz=rotation_to_wxyz(cur_right_pose[:3, :3]))
    
    rate = RateLimiter(frequency=100, warn=True)
    while True:
        # left
        cur_left_qpos = h_robot_left._urdf.cfg.copy()
        h_left_qpos_text.value = f"{np.round(cur_left_qpos, 3).tolist()}"

        cur_left_pose = left_kine.get_fk(cur_left_qpos)
        h_left_ee_pose_text.value = f"{np.round(cur_left_pose[:3,3], 3).tolist()}," + \
                                    f"{np.round(rotation_to_wxyz(cur_left_pose[:3,:3]), 3).tolist()}"
        
        cur_left_psi = left_kine.calc_arm_angle(cur_left_qpos)
        left_cfg = SRSKinematics.Config.from_qpos(cur_left_qpos, cur_left_psi)
        res1, left_intervals = left_kine.calc_feasible_arm_angle_intervals(cur_left_pose, left_cfg)
        if res1 == KineStatus.OK:
            for interval in left_intervals:
                if interval.contains(cur_left_psi):
                    h_left_arm_angle_text.value = f"{np.round(cur_left_psi, 3)} ∈{interval}"
        else:
            h_left_arm_angle_text.value = "Error"

        goal_left_pose = pos_wxyz_to_homogeneous(h_ee_goal_left.position, 
                                                 h_ee_goal_left.wxyz)
        res, left_qpos = left_kine.get_ik(goal_left_pose, left_cfg, cur_left_qpos)
        if res == KineStatus.OK:
            h_robot_left.update_cfg(left_qpos)

        # right

        rate.sleep()    