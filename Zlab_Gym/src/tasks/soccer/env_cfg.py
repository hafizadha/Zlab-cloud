"""Single-robot kicking and open-field soccer on flat ground."""

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from src.tasks.velocity.config.z01_std_add_head.env_cfgs import (
    z01_std_add_head_flat_env_cfg,
)

from . import mdp


def _ball_spec() -> mujoco.MjSpec:
    # Match the mass and contact settings of the RoboCup soccer ball.
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="soccer_ball")
    body.add_freejoint(name="ball_free_joint")
    body.add_geom(
        name="ball_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(mdp.BALL_RADIUS, 0.0, 0.0),
        mass=0.4,
        rgba=(0.2, 0.6, 0.9, 1.0),
        condim=6,
        friction=(0.8, 0.005, 0.004),
        solref=(0.02, 1.0),
        solimp=(0.9, 0.95, 0.001, 0.5, 2.0),
        priority=1,
    )
    return spec


def _add_goal_geometry(
    spec: mujoco.MjSpec, x: float, width: float, height: float
) -> None:
    goal = spec.worldbody.add_body(name="soccer_goal", pos=(x, 0, 0))
    for name, endpoints in (
        ("left_post", (0, width, 0, 0, width, height)),
        ("right_post", (0, -width, 0, 0, -width, height)),
        ("crossbar", (0, -width, height, 0, width, height)),
    ):
        goal.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=endpoints,
            size=(mdp.POST_RADIUS, 0.0, 0.0),
            rgba=(0.9, 0.9, 0.9, 1.0),
        )
    goal.add_geom(
        name="goal_line",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(0, 0, 0.002),
        size=(0.02, width, 0.001),
        contype=0,
        conaffinity=0,
        rgba=(1.0, 1.0, 1.0, 1.0),
    )


def _add_kick_goal(spec: mujoco.MjSpec) -> None:
    _add_goal_geometry(spec, mdp.GOAL_X, mdp.GOAL_HALF_WIDTH, mdp.GOAL_HEIGHT)


def _add_field(spec: mujoco.MjSpec) -> None:
    x, y = mdp.FIELD_GOAL_X, mdp.FIELD_HALF_WIDTH
    _add_goal_geometry(spec, x, mdp.FIELD_GOAL_HALF_WIDTH, mdp.FIELD_GOAL_HEIGHT)
    for name, pos, size in (
        ("left_touchline", (0, y, 0.002), (x, 0.025, 0.001)),
        ("right_touchline", (0, -y, 0.002), (x, 0.025, 0.001)),
        ("own_endline", (-x, 0, 0.002), (0.025, y, 0.001)),
        ("opposing_endline", (x, 0, 0.002), (0.025, y, 0.001)),
    ):
        spec.worldbody.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos,
            size=size,
            contype=0,
            conaffinity=0,
            rgba=(0.9, 0.9, 0.9, 1.0),
        )


def soccer_env_cfg(
    play: bool = False, approach: bool = False, sprint: bool = False
) -> ManagerBasedRlEnvCfg:
    cfg = z01_std_add_head_flat_env_cfg(play=play)
    cfg.episode_length_s = 18.0 if approach else 5.0
    goal_x = mdp.FIELD_GOAL_X if approach else mdp.GOAL_X
    goal_params = {"goal_x": goal_x}
    if approach:
        cfg.viewer.distance = 6.0
        cfg.viewer.elevation = -25.0
    cfg.scene.env_spacing = 0.0  # Independent Warp worlds share one goal layout.
    cfg.scene.entities["ball"] = EntityCfg(
        spec_fn=_ball_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(0.38, -0.10, mdp.BALL_RADIUS)),
    )
    cfg.scene.spec_fn = _add_field if approach else _add_kick_goal
    foot_geoms = r".*_foot[1-7]_collision"
    ball_match = ContactMatch(mode="geom", pattern="ball_geom", entity="ball")
    cfg.scene.sensors += (
        ContactSensorCfg(
            name="foot_ball_contact",
            primary=ContactMatch(mode="geom", pattern=foot_geoms, entity="robot"),
            secondary=ball_match,
            fields=("force",),
            reduce="netforce",
            history_length=cfg.decimation,
        ),
        ContactSensorCfg(
            name="body_ball_contact",
            primary=ContactMatch(
                mode="geom",
                pattern=".*_collision",
                entity="robot",
                exclude=(foot_geoms,),
            ),
            secondary=ball_match,
            fields=("force",),
            reduce="netforce",
            history_length=cfg.decimation,
        ),
    )

    cfg.commands.clear()
    cfg.curriculum.clear()
    for obs in cfg.observations.values():
        # Keep this term in the same position as the bundled walking actor's twist.
        obs.terms["command"] = ObservationTermCfg(
            func=mdp.approach_command,
            params={**goal_params, "field": approach, "sprint": sprint},
        )
        obs.terms["soccer"] = ObservationTermCfg(
            func=mdp.soccer_state, params=goal_params
        )

    # Use repeatable standing resets while learning contact and weight transfer.
    cfg.events = {
        name: cfg.events[name] for name in ("reset_base", "reset_robot_joints")
    }
    cfg.events["reset_base"].params.update(
        pose_range={"yaw": (-0.12, 0.12)} if approach else {},
        velocity_range={},
    )
    cfg.events["reset_robot_joints"].params.update(
        position_range=(0.0, 0.0), velocity_range=(0.0, 0.0)
    )
    cfg.events["reset_ball"] = EventTermCfg(
        func=mdp.reset_ball,
        mode="reset",
        params={
            "x_range": (-2.40, 2.50) if approach else (0.32, 0.42),
            "y_range": (-2.25, 2.25) if approach else (-0.14, -0.06),
            "spawn_clearance": 0.55 if approach else 0.0,
            "speed_range": (0.0, 0.0),
        },
    )

    cfg.rewards = {
        name: cfg.rewards[name]
        for name in ("upright", "action_rate_l2", "dof_pos_limits", "self_collisions")
    }
    cfg.rewards["upright"].weight = 0.5
    cfg.rewards.update(
        alignment=RewardTermCfg(
            func=mdp.approach_alignment, weight=0.4, params=goal_params
        ),
        kick_swing=RewardTermCfg(
            func=mdp.kick_swing,
            weight=5.0,
            params={
                **goal_params,
                "asset_cfg": SceneEntityCfg("robot", site_names=("right_foot",)),
            },
        ),
        ball_toward_goal=RewardTermCfg(
            func=mdp.ball_toward_goal, weight=2.0, params=goal_params
        ),
        body_ball_contact=RewardTermCfg(
            func=mdp.ball_contact,
            weight=-15.0,
            params={"sensor_name": "body_ball_contact"},
        ),
        goal=RewardTermCfg(func=mdp.goal_bonus, weight=10.0),
    )
    goal_dimensions = (
        {"half_width": mdp.FIELD_GOAL_HALF_WIDTH, "height": mdp.FIELD_GOAL_HEIGHT}
        if approach
        else {}
    )
    bounds = (
        {"min_x": -mdp.FIELD_GOAL_X, "half_width": mdp.FIELD_HALF_WIDTH}
        if approach
        else {}
    )
    cfg.terminations.update(
        goal=TerminationTermCfg(
            func=mdp.GoalCrossing, params={**goal_params, **goal_dimensions}
        ),
        ball_out=TerminationTermCfg(
            func=mdp.ball_out, params={**goal_params, **bounds}
        ),
    )
    if approach:
        cfg.rewards["approach_progress"] = RewardTermCfg(
            func=mdp.approach_progress, weight=1.5, params=goal_params
        )
    if sprint:
        cfg.rewards["sprint_velocity_tracking"] = RewardTermCfg(
            func=mdp.sprint_velocity_tracking, weight=2.0, params=goal_params
        )
    return cfg
