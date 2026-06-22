from types import SimpleNamespace

import numpy as np

from channel_models import MCSCodeset
from schedulers import ProportionalFair


def test_mcs_selection_returns_selected_mcs_nominal_rate():
    codeset = MCSCodeset()

    for snr_db in np.linspace(-5.0, 30.0, 36):
        mcs, bits_per_symbol = codeset.mcs_rate_vs_error(snr_db, 0.1)
        assert np.isclose(bits_per_symbol, codeset.nominal_rate(mcs))


def test_pf_realized_rx_probability_respects_error_target():
    codeset = MCSCodeset()
    scheduler = ProportionalFair(codeset, granularity=1)
    ue = SimpleNamespace(
        th=1.0,
        queue=100_000,
        e_snr=20.0,
        snr=np.full(8, 20.0, dtype=float),
        prbs=0,
        p=0.0,
        bits=0,
        useful_prbs=0,
        wasted_prbs=0,
        mcs=None,
        spectral_efficiency=0.0,
        effective_sinr_db=float("nan"),
    )

    scheduler.allocate([ue], n_prb=8, error_bound=0.1)

    assert ue.prbs > 0
    assert ue.p >= 0.90
    assert np.isfinite(ue.effective_sinr_db)
    assert np.isclose(ue.spectral_efficiency, codeset.nominal_rate(ue.mcs))


def _scheduler_ue(slice_type):
    return SimpleNamespace(
        slice_type=slice_type,
        th=1.0,
        queue=100_000,
        e_snr=10.0,
        snr=np.full(8, 10.0, dtype=float),
        prbs=0,
        p=0.0,
        bits=0,
        useful_prbs=0,
        wasted_prbs=0,
        mcs=None,
        spectral_efficiency=0.0,
        effective_sinr_db=float("nan"),
        mcs_codeset_name="default",
    )


def test_urllc_uses_dedicated_codeset_while_other_slices_use_default():
    default_codeset = MCSCodeset()
    urllc_codeset = MCSCodeset("datasets/mcs_codeset_urllc.csv")
    scheduler = ProportionalFair(
        default_codeset,
        granularity=1,
        mcs_codesets_by_slice={"URLLC": urllc_codeset},
    )

    for slice_type, expected_codeset, expected_name in (
        ("eMBB", default_codeset, "default"),
        ("URLLC", urllc_codeset, "URLLC"),
        ("mMTC", default_codeset, "default"),
    ):
        ue = _scheduler_ue(slice_type)
        scheduler.allocate([ue], n_prb=8, error_bound=0.1)
        expected_mcs, expected_rate = expected_codeset.mcs_rate_vs_error(10.0, 0.1)
        assert ue.mcs == expected_mcs
        assert np.isclose(ue.spectral_efficiency, expected_rate)
        assert ue.mcs_codeset_name == expected_name
