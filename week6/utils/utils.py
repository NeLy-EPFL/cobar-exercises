import numpy as np
from flygym.examples.locomotion import CPGNetwork, PreprogrammedSteps
from flygym.preprogrammed import all_leg_dofs
from flygym import SingleFlySimulation, Fly
from flygym.arena import BaseArena
from gymnasium import spaces

from typing import List
from pathlib import Path

_tripod_phase_biases = np.pi * np.array(
    [
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
    ]
)
_tripod_coupling_weights = (_tripod_phase_biases > 0) * 10

right_turns_descending = np.array([1.2, -0.8])
left_turns_descending = np.array([-0.8, 1.2])

class TurningController(SingleFlySimulation):
    """
    This class implements a controller that uses a CPG network to generate
    leg movements and uses a set of sensory-based rules to correct for
    stumbling and retraction. The controller also receives a 2D descending
    input to modulate the amplitudes and frequencies of the CPGs to
    accomplish turning.

    Notes
    -----
    Please refer to the `"MPD Task Specifications" page
    <https://neuromechfly.org/api_ref/mdp_specs.html#hybrid-turning-controller-hybridturningcontroller>`_
    of the API references for the detailed specifications of the action
    space, the observation space, the reward, the "terminated" and
    "truncated" flags, and the "info" dictionary.

    Parameters
    ----------
    fly : Fly
        The fly object to be simulated.
    preprogrammed_steps : PreprogrammedSteps, optional
        Preprogrammed steps to be used for leg movement.
    intrinsic_freqs : np.ndarray, optional
        Intrinsic frequencies of the CPGs. See ``CPGNetwork`` for
        details.
    intrinsic_amps : np.ndarray, optional
        Intrinsic amplitudes of the CPGs. See ``CPGNetwork`` for
        details.
    phase_biases : np.ndarray, optional
        Phase biases of the CPGs. See ``CPGNetwork`` for details.
    coupling_weights : np.ndarray, optional
        Coupling weights of the CPGs. See ``CPGNetwork`` for details.
    convergence_coefs : np.ndarray, optional
        Convergence coefficients of the CPGs. See ``CPGNetwork`` for
        details.
    init_phases : np.ndarray, optional
        Initial phases of the CPGs. See ``CPGNetwork`` for details.
    init_magnitudes : np.ndarray, optional
        Initial magnitudes of the CPGs. See ``CPGNetwork`` for details.
    seed : int, optional
        Seed for the random number generator.
    amplitude_range: tuple, optional
        Range of descending signals that can be applied to the CPGs.
    init_control_mode : str, optional
        Initial control mode. Can be "CPG", "Single", or "Tripod".
    leg_step_time : float, optional
        Time taken to step a leg in seconds.
    **kwargs
        Additional keyword arguments to be passed to
        ``SingleFlySimulation.__init__``.
    """

    def __init__(
        self,
        fly: Fly,
        preprogrammed_steps=None,
        intrinsic_freqs=np.ones(6) * 36,  # np.ones(6) * 12,
        intrinsic_amps=np.ones(6) * 6,
        phase_biases=_tripod_phase_biases,
        coupling_weights=_tripod_coupling_weights,
        convergence_coefs=np.ones(6) * 20,
        init_phases=None,
        init_magnitudes=None,
        amplitude_range=(-0.5, 1.5),
        seed=0,
        init_control_mode="CPG",
        leg_step_time=0.025,
        **kwargs,
    ):
        # Check if we have the correct list of actuated joints
        if fly.actuated_joints != all_leg_dofs:
            raise ValueError(
                "``HybridTurningController`` requires a specific set of DoFs, namely "
                "``flygym.preprogrammed.all_leg_dofs``, to be actuated. A different "
                "set of DoFs was provided."
            )

        # Initialize core NMF simulation
        super().__init__(fly=fly, **kwargs)
        if preprogrammed_steps is None:
            preprogrammed_steps = PreprogrammedSteps()
        self.preprogrammed_steps = preprogrammed_steps
        self.intrinsic_freqs = intrinsic_freqs
        self.intrinsic_amps = intrinsic_amps
        self.phase_biases = phase_biases
        self.coupling_weights = coupling_weights
        self.convergence_coefs = convergence_coefs

        # Define action and observation spaces
        self.action_space = spaces.Box(*amplitude_range, shape=(2,))

        # Initialize CPG network
        self.cpg_network = CPGNetwork(
            timestep=self.timestep,
            intrinsic_freqs=intrinsic_freqs,
            intrinsic_amps=intrinsic_amps,
            coupling_weights=coupling_weights,
            phase_biases=phase_biases,
            convergence_coefs=convergence_coefs,
            init_phases=init_phases,
            init_magnitudes=init_magnitudes,
            seed=seed,
        )

        self.prev_control_mode = init_control_mode

        self.leg_phases = np.zeros(6)
        self.step_direction = np.zeros(6)
        self.phase_increment = self.timestep / leg_step_time * 2 * np.pi

        self.tripod_map = {"LF": 0, "LM": 1, "LH": 0, "RF": 1, "RM": 0, "RH": 1}
        self.tripod_phases = np.zeros(2)

    def reset(self, seed=None, init_phases=None, init_magnitudes=None, **kwargs):
        """
        Reset the simulation.

        Parameters
        ----------
        seed : int, optional
            Seed for the random number generator. If None, the simulation
            is re-seeded without a specific seed. For reproducibility,
            always specify a seed.
        init_phases : np.ndarray, optional
            Initial phases of the CPGs. See ``CPGNetwork`` for details.
        init_magnitudes : np.ndarray, optional
            Initial magnitudes of the CPGs. See ``CPGNetwork`` for details.
        **kwargs
            Additional keyword arguments to be passed to
            ``SingleFlySimulation.reset``.

        Returns
        -------
        np.ndarray
            Initial observation upon reset.
        dict
            Additional information.
        """
        obs, info = super().reset(seed=seed)
        self.cpg_network.random_state = np.random.RandomState(seed)
        self.cpg_network.intrinsic_amps = self.intrinsic_amps
        self.cpg_network.intrinsic_freqs = self.intrinsic_freqs
        self.cpg_network.reset(init_phases, init_magnitudes)

        self.leg_phases = np.zeros(6)
        self.step_direction = np.zeros(6)

        return obs, info

    def get_cpg_joint_angles(self, action):
        amps = np.repeat(np.abs(action[:, np.newaxis]), 3, axis=1).ravel()
        freqs = self.intrinsic_freqs.copy()
        freqs[:3] *= 1 if action[0] > 0 else -1
        freqs[3:] *= 1 if action[1] > 0 else -1
        self.cpg_network.intrinsic_amps = amps
        self.cpg_network.intrinsic_freqs = freqs

        self.cpg_network.step()

        joints_angles = []
        adhesion_onoff = []
        for i, leg in enumerate(self.preprogrammed_steps.legs):
            # get target angles from CPGs and apply correction
            my_joints_angles = self.preprogrammed_steps.get_joint_angles(
                leg,
                self.cpg_network.curr_phases[i],
                self.cpg_network.curr_magnitudes[i],
            )
            joints_angles.append(my_joints_angles)

            # get adhesion on/off signal
            my_adhesion_onoff = self.preprogrammed_steps.get_adhesion_onoff(
                leg, self.cpg_network.curr_phases[i]
            )

            # No adhesion in stumbling or retracted
            adhesion_onoff.append(my_adhesion_onoff)

        return {
            "joints": np.array(np.concatenate(joints_angles)),
            "adhesion": np.array(adhesion_onoff).astype(int),
        }

    def step(self, action):
        """Step the simulation forward one timestep.

        Parameters
        ----------
        action : np.ndarray
            Array of shape (2,) containing descending signal encoding
            turning.
        activated_legs : np.ndarray
            Array of shape (6,) containing the legs we want to activate
        """

        joints_action = self.get_cpg_joint_angles(action)
        obs, reward, terminated, truncated, info = super().step(joints_action)
        info.update(joints_action)  # add lower-level action to info
        return obs, reward, terminated, truncated, info


class BaseBallIntegrationArena(BaseArena):
    def __init__(
        self,
        size: tuple[float, float] = (100, 100),
        friction: tuple[float, float, float] = (1, 0.005, 0.0001),
        ground_alpha: float = 1.0,
        seed: int = 0, 
        cues_radius: float = 1.0,
        cues_height: float = 300,
        cues_radius_interval: List[int] = [50, 60],
        n_cues: int = 20
    ):
        super().__init__()
        self.add_baseball_floor(size)

        np.random.seed(seed)
        self.add_visual_cues(cues_radius, cues_height, cues_radius_interval, n_cues)

        self.friction = friction
    
    def add_visual_cues(self, cues_radius, cues_height, cues_radius_interval, n_cues):
        self.cues = []
        for i in range(n_cues):
            r = np.random.uniform(*cues_radius_interval)
            theta = np.random.uniform(0, 2*np.pi)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            rgb = np.random.uniform(0, 1, 3)
            cue = self.add_cue(x, y, cues_radius, cues_height, rgb, i)
            self.cues.append(cue)

    def add_cue(self, x, y, r, h, rgb, i):
        # Note that here we are not attaching the cue to a specific 
        cue = self.root_element.worldbody.add(
                "geom",
                type="cylinder",
                name =f"cue_{i}",
                pos = (x, y, 0),
                size=(r, h / 2),
                rgba=(0.25, 0.25, 0.25, 1),
                contype=0, # This can be used to set contacts see: https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom-contype
                # Contacts are taking up a lot of computation time, so disabling them for objects that we do not need to interact with is good practice
            )

    def add_baseball_floor(self, size: tuple):
        # path of current file
        current_file = Path(__file__).resolve()
        baseball_texture = current_file.parent.parent / "source/ground-texture.png"


        baseball_texture = self.root_element.asset.add(
            "texture",
            type="2d",
            file=str(baseball_texture),
        )
        grid = self.root_element.asset.add(
            "material",
            name="grid",
            texture=baseball_texture,
            texrepeat=(1, 1),
        )
        self.root_element.worldbody.add(
            "geom",
            type="plane",
            name="ground",
            material=grid,
            size=[*size, 1],
            friction=self.friction,
            conaffinity=0,
        )

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return rel_pos, rel_angle

    def _get_max_floor_height(self) -> float:
        geom = self.root_element.find("geom", "ground")
        try:
            plane_height = geom.pos[2]
        except TypeError:
            plane_height = 0.0

        return plane_height

class SimpleHeadStabilisedFly(Fly):
    def __init__(self, **kwargs):
        super().__init__(
            contact_sensor_placements=[
                f"{side}{pos}{segment}"
                for side in "LR"
                for pos in "FMH"
                for segment in ["Tarsus5"]
            ],
            self_collisions="none",
            floor_collisions=[
                f"{side}{pos}{dof}"
                for side in "LR"
                for pos in "FMH"
                for dof in ["Tarsus5"]
            ],
            xml_variant="seqik_simple",
            head_stabilization_model="thorax",
            neck_kp=1000,
            spawn_orientation=(0, 0, -np.pi / 2),
            enable_adhesion=True,
            **kwargs,
        )

        fly_head_body = self.model.worldbody.find("body", "Head")
        fly_head_body.add(
            "joint",
            **{
                "name": "joint_Head_yaw",
                "class": "nmf",
                "type": "hinge",
                "pos": "0 0 0",
                "axis": "1 0 0",
                "stiffness": str(self.neck_stiffness),
                "springref": "0.0",
                "damping": str(self.non_actuated_joint_damping),
                "frictionloss": "0.0",
            },
        )
        fly_head_body.add(
            "joint",
            **{
                "name": "joint_Head",
                "class": "nmf",
                "type": "hinge",
                "pos": "0 0 0",
                "axis": "0 1 0",
                "stiffness": str(self.neck_stiffness),
                "springref": "0.0",
                "damping": str(self.non_actuated_joint_damping),
                "frictionloss": "0.0",
            },
        )
        fly_head_body.add(
            "joint",
            **{
                "name": "joint_Head_roll",
                "class": "nmf",
                "type": "hinge",
                "pos": "0 0 0",
                "axis": "0 0 1",
                "stiffness": str(self.neck_stiffness),
                "springref": "0.0",
                "damping": str(self.non_actuated_joint_damping),
                "frictionloss": "0.0",
            },
        )
        self.model.visual.map.zfar = 10