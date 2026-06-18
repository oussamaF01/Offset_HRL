#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np

from scenario_creator import create_multignb_env


def run_sanity_checks() -> None:
    rng = np.random.default_rng(123)
    env = create_multignb_env(
        rng=rng,
        n=4,
        n_gnbs=1,
        L1_level=False,
        use_sumo_mobility=False,
        mobility_dt=0.0,
        radio_substeps=1,
        max_episode_steps=5,
    )

    try:
        gnb_id = int(env.gnbs[0].id)

        for l1 in env.gnbs[0].slices_l1:
            if env.normalize_slice_type(getattr(l1, "type", "")) == "eMBB":
                l1.n_prbs = 50

        assert env.estimate_slice_load(gnb_id, "SliceL1eMBB") == 0.0

        ue_ids = [
            env.add_ue(x=0.0, y=0.0, vx=0.0, vy=0.0, slice_type="eMBB")
            for _ in range(10)
        ]
        for ue_id in ue_ids:
            ue = env.get_ue(ue_id)
            ue.connected = True
            ue.serving_gnb = gnb_id
            ue.prbs = 1

        assert env.get_slice_ue_count(gnb_id, "eMBB") == 10
        assert np.isclose(env.get_slice_used_prbs(gnb_id, "eMBB"), 10)
        assert np.isclose(env.estimate_slice_load(gnb_id, "eMBB"), 0.2)

        for ue_id in ue_ids:
            env.get_ue(ue_id).prbs = 6

        assert env.get_slice_used_prbs(gnb_id, "eMBB") == 60
        assert env.estimate_slice_load(gnb_id, "eMBB") == 1.0
        assert env.estimate_slice_load(gnb_id, "eMBB") != env.get_slice_ue_count(gnb_id, "eMBB")

        n_keys = len(env.gnbs) * len(env._configured_slice_types())
        obs_with_counts = env.get_global_agent_observation(include_ue_counts=True)
        obs_without_counts = env.get_global_agent_observation(include_ue_counts=False)
        assert obs_with_counts.shape == (n_keys * 3,)
        assert obs_without_counts.shape == (n_keys * 2,)

        print("global slice PRB load sanity checks passed")
    finally:
        env.close()


if __name__ == "__main__":
    run_sanity_checks()
