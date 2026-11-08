"""Link budgets, airtime, duty cycle, collision load and battery model per radio technology.

Numbers are datasheet-class / standard values, not measurements. See README "What is not verified".
Uplink only: meters transmit, gateways (collectors) listen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

LORA_BW_HZ = 125_000.0
# Typical sensitivity at 125 kHz (sub-GHz LoRa modem datasheet class), dBm
LORA_SENS = {7: -123.0, 8: -126.0, 9: -129.0, 10: -132.0, 11: -134.5, 12: -137.0}


def lora_airtime_s(sf: int, payload_bytes: int, bw_hz: float = LORA_BW_HZ, cr: int = 1,
                   preamble: int = 8, explicit_header: bool = True, crc: bool = True) -> float:
    """Time on air of a LoRa frame (standard LoRa time-on-air formula). payload_bytes = full PHY payload."""
    if not 6 <= sf <= 12:
        raise ValueError("SF must be 6..12")
    if payload_bytes < 0:
        raise ValueError("payload must be >= 0")
    tsym = (2**sf) / bw_hz
    de = 1 if (sf >= 11 and bw_hz <= 125_000) else 0
    h = 0 if explicit_header else 1
    num = 8 * payload_bytes - 4 * sf + 28 + (16 if crc else 0) - 20 * h
    n_pay = 8 + max(math.ceil(num / (4 * (sf - 2 * de))) * (cr + 4), 0)
    return (preamble + 4.25) * tsym + n_pay * tsym


@dataclass(frozen=True)
class Level:
    """One data rate / spreading factor of a technology."""

    name: str
    sens_dbm: float
    airtime_s: float
    rx_window_s: float = 0.0  # receiver-on time per uplink (class A windows), 0 for uplink-only


@dataclass(frozen=True)
class Tech:
    name: str
    freq_mhz: float
    tx_dbm: float
    node_gain_dbi: float
    gw_gain_dbi: float
    gw_loss_db: float
    levels: tuple[Level, ...]  # ordered fastest -> most sensitive
    channels: int  # parallel uplink channels for collision estimate
    duty_cycle: float  # regulatory limit per sub-band (fraction)
    tx_ma: float
    rx_ma: float
    sleep_ua: float = 5.0
    self_discharge_ua: float = 3.0  # ~1 %/year of a 2.6 Ah lithium cell, counted as a constant drain
    battery_mah: float = 2600.0  # one 3.6 V AA-size lithium cell
    battery_derate: float = 0.8
    volt: float = 3.6

    def mapl_db(self, level: Level, fade_margin_db: float) -> float:
        """Maximum allowed path loss (before penetration) for a level."""
        return (self.tx_dbm + self.node_gain_dbi + self.gw_gain_dbi - self.gw_loss_db
                - level.sens_dbm - fade_margin_db)

    def mapl_vector(self, fade_margin_db: float) -> np.ndarray:
        return np.array([self.mapl_db(lv, fade_margin_db) for lv in self.levels])

    def energy_per_message_mj(self, level: Level) -> float:
        return self.volt * (self.tx_ma * level.airtime_s + self.rx_ma * level.rx_window_s)

    def battery_years(self, level: Level, period_s: float) -> float:
        """Average-current battery life for one message every period_s (sleep current included)."""
        e_msg_mc = self.tx_ma * level.airtime_s + self.rx_ma * level.rx_window_s  # mA*s
        avg_ma = e_msg_mc / period_s + (self.sleep_ua + self.self_discharge_ua) / 1000.0
        hours = self.battery_mah * self.battery_derate / avg_ma
        return hours / (24 * 365)

    def duty_used(self, level: Level, period_s: float) -> float:
        """Fraction of the regulatory duty-cycle budget one meter uses."""
        return (level.airtime_s / period_s) / self.duty_cycle


def lorawan_eu868(app_payload_bytes: int = 12) -> Tech:
    phy = app_payload_bytes + 13  # LoRaWAN MHDR + FHDR + FPort + MIC
    levels = []
    for sf in range(7, 13):
        tsym = (2**sf) / LORA_BW_HZ
        # class A: two receive windows, each ~ 8 symbols of preamble detection
        levels.append(Level(f"SF{sf}", LORA_SENS[sf], lora_airtime_s(sf, phy), rx_window_s=2 * 8 * tsym))
    return Tech("lorawan", 868.1, 14.0, -2.0, 5.0, 1.0, tuple(levels), channels=3, duty_cycle=0.01,
                tx_ma=44.0, rx_ma=11.0)


def _wmbus(name: str, sens: float, bits_on_air: float, bitrate: float) -> Tech:
    airtime = bits_on_air / bitrate
    return Tech(name, 868.95, 14.0, -2.0, 5.0, 1.0, (Level(name.upper(), sens, airtime),),
                channels=1, duty_cycle=0.001, tx_ma=35.0, rx_ma=0.0)  # 868.7-869.2 MHz sub-band: 0.1 %


def wmbus_t1(frame_bytes: int = 50) -> Tech:
    """EN 13757-4 mode T1: 100 kchip/s, 3-out-of-6 coding (x1.5), ~ -105 dBm typical receiver."""
    return _wmbus("wmbus_t1", -105.0, frame_bytes * 8 * 1.5 + 48, 100_000.0)


def wmbus_c1(frame_bytes: int = 50) -> Tech:
    """EN 13757-4 mode C1: 100 kbit/s NRZ, ~ -102 dBm typical receiver."""
    return _wmbus("wmbus_c1", -102.0, frame_bytes * 8 + 48, 100_000.0)


def mioty(airtime_s: float = 0.7, sens_dbm: float = -139.0) -> Tech:
    """Parameter set for telegram-splitting LPWAN (ETSI TS 103 357). Values are assumptions."""
    return Tech("mioty", 868.0, 14.0, -2.0, 5.0, 1.0, (Level("mioty", sens_dbm, airtime_s),),
                channels=1, duty_cycle=0.01, tx_ma=44.0, rx_ma=0.0)


def all_techs(app_payload_bytes: int = 12) -> dict[str, Tech]:
    return {t.name: t for t in (lorawan_eu868(app_payload_bytes), wmbus_t1(), wmbus_c1(), mioty())}


def with_sensitivity_offset(tech: Tech, offset_db: float) -> Tech:
    """Shift all sensitivities (negative = better receiver); used for sensitivity checks."""
    return replace(tech, levels=tuple(replace(lv, sens_dbm=lv.sens_dbm + offset_db) for lv in tech.levels))


def aloha_success(offered_g: np.ndarray | float) -> np.ndarray | float:
    """Pure-ALOHA success probability exp(-2G) for G = frames in flight per frame time."""
    return np.exp(-2.0 * np.asarray(offered_g))
