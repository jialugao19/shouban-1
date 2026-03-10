"""Define immutable configuration types for firstboard_topk_v1."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AuctionShapeSbgk:
    """Hold non-linear auction score parameters for the sbgk path."""

    t_min: float
    t_hot: float
    t_max: float


@dataclass(frozen=True)
class AuctionShapeSbdk:
    """Hold non-linear auction score parameters for the sbdk path."""

    t_best: float
    t_max: float


@dataclass(frozen=True)
class OpenBand:
    """Hold the open_ratio target interval for one path."""

    low: float
    high: float


@dataclass(frozen=True)
class StrengthRange:
    """Hold a linear strength score mapping range."""

    r0: float
    r1: float


@dataclass(frozen=True)
class UpperShadowRange:
    """Hold an upper-shadow score mapping range."""

    s0: float
    s1: float


@dataclass(frozen=True)
class VolShrinkageRange:
    """Hold a vol-shrinkage score mapping range."""

    width: float


@dataclass(frozen=True)
class LimitQualityParams:
    """Hold limit-quality scoring parameters."""

    seal_time_norm: int
    open_board_vol_norm: float


@dataclass(frozen=True)
class SbdkWeights:
    """Hold weights for the sbdk path score."""

    w_open_band: float
    w_auction: float
    w_liquidity: float
    w_rp: float
    w_limit_quality: float


@dataclass(frozen=True)
class SbgkWeights:
    """Hold weights for the sbgk path score."""

    w_open_band: float
    w_auction: float
    w_liquidity: float
    w_strength: float
    w_limit_quality: float


@dataclass(frozen=True)
class RzgWeights:
    """Hold weights for the rzg path score."""

    w_open_band: float
    w_auction: float
    w_liquidity: float
    w_strength: float
    w_reversal: float
    w_limit_quality: float


@dataclass(frozen=True)
class StrategyConfig:
    """Declare all inputs, parameters, and output paths for one run."""

    start_date: str
    end_date: str

    trade_cal_path: str
    stock1d_root: str
    stock1m_root: str
    valid_stocks_path: str

    out_root: str

    first_board_lookback: int
    rp_lookback: int
    ma5_lookback: int

    topk_k: int
    score_threshold: float
    dynamic_k_by_score: bool
    filter_limit_up_at_open: bool

    commission_bps: float
    stamp_duty_bps: float
    slippage_bps: float
    initial_cash: float

    open_band_sbdk: OpenBand
    open_band_sbgk: OpenBand
    open_band_rzg: OpenBand

    auction_shape_sbgk: AuctionShapeSbgk
    auction_shape_sbdk: AuctionShapeSbdk

    strength_sbgk: StrengthRange
    strength_rzg: StrengthRange

    upper_shadow_range: UpperShadowRange
    vol_shrinkage_range: VolShrinkageRange

    limit_quality_params: LimitQualityParams

    weights_sbdk: SbdkWeights
    weights_sbgk: SbgkWeights
    weights_rzg: RzgWeights
