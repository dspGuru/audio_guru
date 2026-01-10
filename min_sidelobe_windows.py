"""
Albrecht Minimum Sidelobe Cosine-Sum Windows.

Data transcribed from Table 1 and Table 2 of "Tailoring of Minimum Sidelobe
Cosine-Sum Windows for High-Resolution  Measurements" by Hans-Helge Albrecht,
published in The Open Signal Processing Journal, 2010, 3, pages 20-29.
"""

from typing import Dict, Tuple, TypedDict

import numpy as np
from scipy import signal


from dataclasses import dataclass


@dataclass
class WindowParams:
    """
    Parameters for a minimum sidelobe cosine-sum window.

    :var highest_sidelobe_level_db: The highest sidelobe level in dB.
    :vartype highest_sidelobe_level_db: float
    :var coherent_gain_db: The coherent gain in dB.
    :vartype coherent_gain_db: float
    :var scallop_loss_db: The scallop loss in dB.
    :vartype scallop_loss_db: float
    :var enbw_bins: The equivalent noise bandwidth in bins.
    :vartype enbw_bins: float
    :var bandwidth_3db_bins: The bandwidth at 3 dB in bins.
    :vartype bandwidth_3db_bins: float
    :var bandwidth_6db_bins: The bandwidth at 6 dB in bins.
    :vartype bandwidth_6db_bins: float
    """

    highest_sidelobe_level_db: float
    coherent_gain_db: float
    scallop_loss_db: float
    enbw_bins: float
    bandwidth_3db_bins: float
    bandwidth_6db_bins: float


class WindowData(TypedDict):
    coeffs: Tuple[float, ...]
    params: WindowParams


# Minimum Sidelobe Windows (2-term to 11-term)
MINIMUM_SIDELOBE_WINDOWS: Dict[int, WindowData] = {
    2: {
        "coeffs": (5.383553946707251e-001, 4.616446053292749e-001),  # A0  # A1
        "params": {
            "highest_sidelobe_level_db": 43.187,
            "coherent_gain_db": 5.37862,
            "scallop_loss_db": 1.73868,
            "enbw_bins": 1.36766,
            "bandwidth_3db_bins": 1.30550,
            "bandwidth_6db_bins": 1.81884,
        },
    },
    3: {
        "coeffs": (
            4.243800934609435e-001,  # A0
            4.973406350967378e-001,  # A1
            7.827927144231873e-002,  # A2
        ),
        "params": {
            "highest_sidelobe_level_db": 71.482,
            "coherent_gain_db": 7.44490,
            "scallop_loss_db": 1.13525,
            "enbw_bins": 1.70371,
            "bandwidth_3db_bins": 1.61612,
            "bandwidth_6db_bins": 2.26377,
        },
    },
    4: {
        "coeffs": (
            3.635819267707608e-001,  # A0
            4.891774371450171e-001,  # A1
            1.365995139786921e-001,  # A2
            1.064112210553003e-002,  # A3
        ),
        "params": {
            "highest_sidelobe_level_db": 98.173,
            "coherent_gain_db": 8.78795,
            "scallop_loss_db": 0.85056,
            "enbw_bins": 1.97611,
            "bandwidth_3db_bins": 1.86875,
            "bandwidth_6db_bins": 2.62431,
        },
    },
    5: {
        "coeffs": (
            3.232153788877343e-001,  # A0
            4.714921439576260e-001,  # A1
            1.755341299601972e-001,  # A2
            2.849699010614994e-002,  # A3
            1.261357088292677e-003,  # A4
        ),
        "params": {
            "highest_sidelobe_level_db": 125.427,
            "coherent_gain_db": 9.81016,
            "scallop_loss_db": 0.68006,
            "enbw_bins": 2.21535,
            "bandwidth_3db_bins": 2.09137,
            "bandwidth_6db_bins": 2.94118,
        },
    },
    6: {
        "coeffs": (
            2.935578950102797e-001,  # A0
            4.519357723474506e-001,  # A1
            2.014164714263962e-001,  # A2
            4.792610922105837e-002,  # A3
            5.026196426859393e-003,  # A4
            1.375555679558877e-004,  # A5
        ),
        "params": {
            "highest_sidelobe_level_db": 153.566,
            "coherent_gain_db": 10.64612,
            "scallop_loss_db": 0.56526,
            "enbw_bins": 2.43390,
            "bandwidth_3db_bins": 2.29514,
            "bandwidth_6db_bins": 3.23077,
        },
    },
    7: {
        "coeffs": (
            2.712203605850388e-001,  # A0
            4.334446123274422e-001,  # A1
            2.180041228929303e-001,  # A2
            6.578534329560609e-002,  # A3
            1.076186730534183e-002,  # A4
            7.700127105808265e-004,  # A5
            1.368088305992921e-005,  # A6
        ),
        "params": {
            "highest_sidelobe_level_db": 180.468,
            "coherent_gain_db": 11.33355,
            "scallop_loss_db": 0.48523,
            "enbw_bins": 2.63025,
            "bandwidth_3db_bins": 2.47830,
            "bandwidth_6db_bins": 3.49095,
        },
    },
    8: {
        "coeffs": (
            2.533176817029088e-001,  # A0
            4.163269305810218e-001,  # A1
            2.288396213719708e-001,  # A2
            8.157508425925879e-002,  # A3
            1.773592450349622e-002,  # A4
            2.096702749032688e-003,  # A5
            1.067741302205525e-004,  # A6
            1.280702090361482e-006,  # A7
        ),
        "params": {
            "highest_sidelobe_level_db": 207.512,
            "coherent_gain_db": 11.92669,
            "scallop_loss_db": 0.42506,
            "enbw_bins": 2.81292,
            "bandwidth_3db_bins": 2.64883,
            "bandwidth_6db_bins": 3.73304,
        },
    },
    9: {
        "coeffs": [
            2.384331152777942e-001,  # A0
            4.005545348643820e-001,  # A1
            2.358242530472107e-001,  # A2
            9.527918858383112e-002,  # A3
            2.537395516617152e-002,  # A4
            4.152432907505835e-003,  # A5
            3.685604163298180e-004,  # A6
            1.384355593917030e-005,  # A7
            1.161808358932861e-007,  # A8
        ],
        "params": {
            "highest_sidelobe_level_db": 234.734,
            "coherent_gain_db": 12.45267,
            "scallop_loss_db": 0.37780,
            "enbw_bins": 2.98588,
            "bandwidth_3db_bins": 2.81041,
            "bandwidth_6db_bins": 3.96231,
        },
    },
    10: {
        "coeffs": (
            2.257345387130214e-001,  # A0
            3.860122949150963e-001,  # A1
            2.401294214106057e-001,  # A2
            1.070542338664613e-001,  # A3
            3.325916184016952e-002,  # A4
            6.873374952321475e-003,  # A5
            8.751673238035159e-004,  # A6
            6.008598932721187e-005,  # A7
            1.710716472110202e-006,  # A8
            1.027272130265191e-008,  # A9
        ),
        "params": {
            "highest_sidelobe_level_db": 262.871,
            "coherent_gain_db": 12.92804,
            "scallop_loss_db": 0.33950,
            "enbw_bins": 3.15168,
            "bandwidth_3db_bins": 2.96538,
            "bandwidth_6db_bins": 4.18209,
        },
    },
    11: {
        "coeffs": (
            2.151527506679809e-001,  # A0
            3.731348357785249e-001,  # A1
            2.424243358446660e-001,  # A2
            1.166907592689211e-001,  # A3
            4.077422105878731e-002,  # A4
            1.000904500852923e-002,  # A5
            1.639806917362033e-003,  # A6
            1.651660820997142e-004,  # A7
            8.884663168541479e-006,  # A8
            1.938617116029048e-007,  # A9
            8.482485599330470e-010,  # A10
        ),
        "params": {
            "highest_sidelobe_level_db": 289.635,
            "coherent_gain_db": 13.34506,
            "scallop_loss_db": 0.30908,
            "enbw_bins": 3.30480,
            "bandwidth_3db_bins": 3.10851,
            "bandwidth_6db_bins": 4.38506,
        },
    },
}


def min_sidelobe_window(n_terms: int, M: int, sym: bool = True) -> np.ndarray:
    """Return a minimum sidelobe window with the specified number of terms.

    Parameters
    ----------
    n_terms : int
        Number of terms in the window.
    M : int
        Number of points in the window.
    sym : bool, optional
        If True, the window is symmetric. If False, the window is periodic.

    Returns
    -------
    np.ndarray
        The minimum sidelobe window.
    """
    coeffs = MINIMUM_SIDELOBE_WINDOWS[n_terms]["coeffs"]
    return signal.windows.general_cosine(M, coeffs, sym)
