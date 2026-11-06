"""GPS navigation bit, frame, parity, and ephemeris decoding."""

import math
from collections import deque
from statistics import median


class Navigation:
    def __init__(self) -> None:
        self.prompt_signs: list[int] = []
        self.transition_histogram: list[int] = [0] * 20
        self.previous_sign: int | None = None
        self.transition_bit = 0
        self.bit_locked = False
        self.prompt_i_sum = 0.0
        self.navigation_bits: list[int] = []
        self.temporary_bits: list[int] = []
        self.preamble_locked = False
        self.frame_locked = False
        self.frame_bit = 0
        self.costas_inverted = False
        self.preamble_index = 0
        self.words: list[tuple[bool, list[int]]] = []
        self.symbol_ms_count = 0
        self.symbol_valid_count = 0
        self.last_bit = 0
        self.bit_ready = False
        self.navigation_base_index = 0
        self.next_word_index = 0
        self.prompt_power_history: deque[float] = deque(maxlen=101)

    def _invert(self, bits: list[int]) -> list[int]:
        self._validate_bits(bits, "bits")
        return [1 - bit for bit in bits]

    @staticmethod
    def _validate_bits(bits: list[int], name: str) -> None:
        if any(
            not isinstance(bit, int) or isinstance(bit, bool) or bit not in (0, 1)
            for bit in bits
        ):
            raise ValueError(f"{name} must contain only binary integer bits")

    def _gps_parity(
        self, data: list[int], previous_d29: int, previous_d30: int
    ) -> list[int]:
        if len(data) != 24:
            raise ValueError("data must contain exactly 24 bits")
        self._validate_bits(data, "data")
        self._validate_bits([previous_d29, previous_d30], "previous parity bits")
        d = data  # d[0] corresponds to d1, d[23] corresponds to d24

        p1 = (
            previous_d29
            ^ d[0]
            ^ d[1]
            ^ d[2]
            ^ d[4]
            ^ d[5]
            ^ d[9]
            ^ d[10]
            ^ d[11]
            ^ d[12]
            ^ d[13]
            ^ d[16]
            ^ d[17]
            ^ d[19]
            ^ d[22]
        )

        p2 = (
            previous_d30
            ^ d[1]
            ^ d[2]
            ^ d[3]
            ^ d[5]
            ^ d[6]
            ^ d[10]
            ^ d[11]
            ^ d[12]
            ^ d[13]
            ^ d[14]
            ^ d[17]
            ^ d[18]
            ^ d[20]
            ^ d[23]
        )

        p3 = (
            previous_d29
            ^ d[0]
            ^ d[2]
            ^ d[3]
            ^ d[4]
            ^ d[6]
            ^ d[7]
            ^ d[11]
            ^ d[12]
            ^ d[13]
            ^ d[14]
            ^ d[15]
            ^ d[18]
            ^ d[19]
            ^ d[21]
        )

        p4 = (
            previous_d30
            ^ d[1]
            ^ d[3]
            ^ d[4]
            ^ d[5]
            ^ d[7]
            ^ d[8]
            ^ d[12]
            ^ d[13]
            ^ d[14]
            ^ d[15]
            ^ d[16]
            ^ d[19]
            ^ d[20]
            ^ d[22]
        )

        p5 = (
            previous_d30
            ^ d[0]
            ^ d[2]
            ^ d[4]
            ^ d[5]
            ^ d[6]
            ^ d[8]
            ^ d[9]
            ^ d[13]
            ^ d[14]
            ^ d[15]
            ^ d[16]
            ^ d[17]
            ^ d[20]
            ^ d[21]
            ^ d[23]
        )

        p6 = (
            previous_d29
            ^ d[2]
            ^ d[4]
            ^ d[5]
            ^ d[7]
            ^ d[8]
            ^ d[9]
            ^ d[10]
            ^ d[12]
            ^ d[14]
            ^ d[18]
            ^ d[21]
            ^ d[22]
            ^ d[23]
        )

        return [p1, p2, p3, p4, p5, p6]

    def decode_word(
        self, raw_word: list[int], previous_d29: int, previous_d30: int
    ) -> tuple[bool, list[int]]:
        """raw_word: 30 transmitted bits. previous_d29/d30: last two transmitted
        bits of the preceding word, in corrected polarity.
        Returns (parity_ok, data_bits d1..d24)."""
        if len(raw_word) != 30:
            raise ValueError("raw_word must contain exactly 30 bits")
        self._validate_bits(raw_word, "raw_word")
        self._validate_bits([previous_d29, previous_d30], "previous parity bits")
        data = [bit ^ previous_d30 for bit in raw_word[:24]]  # undo D30* on data only
        expected = self._gps_parity(data, previous_d29, previous_d30)
        return raw_word[24:] == expected, data

    def check_word(
        self, raw_word: list[int], previous_d29: int, previous_d30: int
    ) -> bool:
        return self.decode_word(raw_word, previous_d29, previous_d30)[0]

    def process_millisecond(
        self, prn: float | int, prompt_i: float, prompt_power: float, millisecond: int
    ) -> dict[str, float | int | bool | None | list[int]]:
        if not isinstance(millisecond, int) or isinstance(millisecond, bool):
            raise TypeError("millisecond must be an integer")
        if millisecond < 0:
            raise ValueError("millisecond must be non-negative")
        if not isinstance(prompt_i, (int, float)) or isinstance(prompt_i, bool):
            raise TypeError("prompt_i must be a real number")
        if not isinstance(prompt_power, (int, float)) or isinstance(prompt_power, bool):
            raise TypeError("prompt_power must be a real number")
        if not math.isfinite(prompt_i) or not math.isfinite(prompt_power):
            raise ValueError("prompt values must be finite")

        self.prompt_power_history.append(float(prompt_power))
        power_threshold = float(median(self.prompt_power_history))
        self.bit_ready = False
        if prompt_power >= power_threshold:
            current_sign = 1 if prompt_i >= 0 else -1
            if self.previous_sign is not None and self.previous_sign != current_sign:
                self.transition_histogram[millisecond % 20] += 1
            self.previous_sign = current_sign
            self.prompt_signs.append(1 if prompt_i >= 0 else -1)
            self.prompt_i_sum += prompt_i
            self.symbol_valid_count += 1

        if not self.bit_locked and millisecond >= 1000:
            self.transition_bit = self.transition_histogram.index(
                max(self.transition_histogram)
            )
            self.bit_locked = True
            self.prompt_i_sum = 0.0
            self.symbol_ms_count = 0
            self.symbol_valid_count = 0

        if self.bit_locked:
            if millisecond % 20 == self.transition_bit and self.symbol_ms_count:
                if self.symbol_valid_count >= 10:
                    bipolar_bit = 1 if self.prompt_i_sum >= 0 else 0
                    self.last_bit = bipolar_bit
                    self.bit_ready = True
                    self.navigation_bits.append(bipolar_bit)
                    self.temporary_bits.append(bipolar_bit)
                self.prompt_i_sum = 0.0
                self.symbol_valid_count = 0
                self.symbol_ms_count = 0
            self.symbol_ms_count += 1

        if (
            len(self.navigation_bits) >= 30
            and not self.frame_locked
            and not self.preamble_locked
        ):
            gps_preamble = [1, 0, 0, 0, 1, 0, 1, 1]
            index = next(
                (
                    i
                    for i in range(len(self.navigation_bits) - len(gps_preamble) + 1)
                    if self.navigation_bits[i : i + len(gps_preamble)] == gps_preamble
                ),
                -1,  # Default value if no match is found
            )

            if index != -1:
                self.preamble_locked = True
                self.costas_inverted = False
                self.preamble_index = index
                self.temporary_bits = self.navigation_bits[index + 8 :]

            if not self.preamble_locked:
                inverted_preamble = [0, 1, 1, 1, 0, 1, 0, 0]
                index = next(
                    (
                        i
                        for i in range(
                            len(self.navigation_bits) - len(inverted_preamble) + 1
                        )
                        if self.navigation_bits[i : i + len(inverted_preamble)]
                        == inverted_preamble
                    ),
                    -1,
                )

                if index != -1:
                    self.preamble_locked = True
                    self.costas_inverted = True
                    self.preamble_index = index
                    self.temporary_bits = self.navigation_bits[index + 8 :]

        if (
            len(self.temporary_bits) == 300
            and self.preamble_locked
            and not self.frame_locked
        ):
            if self.temporary_bits[-8:] == [1, 0, 0, 0, 1, 0, 1, 1] or (
                self.costas_inverted
                and self.temporary_bits[-8:] == [0, 1, 1, 1, 0, 1, 0, 0]
            ):
                self.frame_locked = True
            else:
                self.preamble_locked = False
            self.temporary_bits.clear()

        if self.frame_locked:
            bits = (
                self._invert(self.navigation_bits)
                if self.costas_inverted
                else self.navigation_bits
            )
            start = (
                self.preamble_index
                + 30 * self.next_word_index
                - self.navigation_base_index
            )
            while start + 30 <= len(bits):
                word = bits[start : start + 30]
                if start < 2:
                    start += 30
                    self.next_word_index += 1
                    continue
                d29, d30 = bits[start - 2], bits[start - 1]
                ok, data = self.decode_word(word, d29, d30)
                self.words.append((ok, data))
                start += 30
                self.next_word_index += 1

        if len(self.prompt_signs) > 20000:
            del self.prompt_signs[:-10000]
        if not self.preamble_locked and len(self.navigation_bits) > 3000:
            del self.navigation_bits[:-1500]
        if (
            self.preamble_locked
            and not self.frame_locked
            and len(self.temporary_bits) > 300
        ):
            del self.temporary_bits[:-300]
        if self.frame_locked and len(self.navigation_bits) > 3000:
            removed = len(self.navigation_bits) - 1500
            del self.navigation_bits[:removed]
            self.navigation_base_index += removed
            self.preamble_index -= removed

        return {
            "bit": self.last_bit,
            "bit_ready": self.bit_ready,
            "frame_locked": self.frame_locked,
            "subframe": 0,
            "clock": 0.0,
            "ephemeris": None,
            "ionosphere": None,
            "navigation_bits": self.navigation_bits,
        }


if __name__ == "__main__":
    from gpsrx.channel_manager import ChannelManager
    from gpsrx.source import IQSource

    with IQSource(4e6, "./data/captures/GPS-L1-2022-03-27.sigmf-data") as source:
        channel_manager = ChannelManager(source)
        results = channel_manager.acquire_all()
        for ms_index in range(15000):
            signal_block = source.get_samples(int(round(source.sampling_rate * 0.001)))
            for channel in channel_manager.channels:
                tracking_results = channel.tracker.process_millisecond(signal_block)
                navigation_results = channel.navigation.process_millisecond(
                    tracking_results["prn"],
                    tracking_results["prompt_i"],
                    tracking_results["prompt_power"],
                    ms_index,
                )
                # if ms_index % 1000 == 0:
                #     print(navigation_results)
