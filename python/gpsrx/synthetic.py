"""Synthetic GPS signal generation for deterministic tests."""

GPS_G2_TAPS = {
    1: (2, 6),
    2: (3, 7),
    3: (4, 8),
    4: (5, 9),
    5: (1, 9),
    6: (2, 10),
    7: (1, 8),
    8: (2, 9),
    9: (3, 10),
    10: (2, 3),
    11: (3, 4),
    12: (5, 6),
    13: (6, 7),
    14: (7, 8),
    15: (8, 9),
    16: (9, 10),
    17: (1, 4),
    18: (2, 5),
    19: (3, 6),
    20: (4, 7),
    21: (5, 8),
    22: (6, 9),
    23: (1, 3),
    24: (4, 6),
    25: (5, 7),
    26: (6, 8),
    27: (7, 9),
    28: (8, 10),
    29: (1, 6),
    30: (2, 7),
    31: (3, 8),
    32: (4, 9),
}


class GPSPRNGenerator:
    def __init__(self, prn_number: int):
        g2_tap1, g2_tap2 = GPS_G2_TAPS[prn_number]
        self.g1 = [1] * 10
        self.g2 = [1] * 10
        self.tap1 = g2_tap1 - 1
        self.tap2 = g2_tap2 - 1

    def clock(self) -> int:
        g1_feedback = self.g1[2] ^ self.g1[9]
        g2_feedback = (
            self.g2[1] ^ self.g2[2] ^ self.g2[5] ^ self.g2[7] ^ self.g2[8] ^ self.g2[9]
        )

        g1_out = self.g1[9]
        g2_out = self.g2[self.tap1] ^ self.g2[self.tap2]

        self.g1 = [g1_feedback] + self.g1[:9]
        self.g2 = [g2_feedback] + self.g2[:9]

        return g1_out ^ g2_out

    def generate_code(self) -> list[int]:
        return [self.clock() for _ in range(1023)]


class SyntheticGPSChannel:
    def __init__(
        self,
        prn_numbers: list[int],
        doppler_shifts: list[float],
        code_phases: list[float],
        carrier_to_noise_ratios: list[float],
    ):
        self.prn_generators = [GPSPRNGenerator(prn) for prn in prn_numbers]

    def generate_baseband_samples(self) -> list[float]:
        samples = []
        for _ in range(1023):
            chip_sum = 0.0
            for gen in self.prn_generators:
                chip_sum = gen.clock()
                voltage = 1.0 if chip_sum == 1 else -1.0
                chip_sum += voltage
            samples.append(chip_sum)
        return samples

if __name__ == "__main__":
    # Example usage
    prn_numbers = [1, 2, 3]  # PRN numbers for the channels
    doppler_shifts = [0.0, 0.0, 0.0]  # Doppler shifts for each channel
    code_phases = [0.0, 0.0, 0.0]  # Code phases for each channel
    carrier_to_noise_ratios = [30.0, 30.0, 30.0]  # C/N ratios for each channel

    synthetic_channel = SyntheticGPSChannel(
        prn_numbers, doppler_shifts, code_phases, carrier_to_noise_ratios
    )
    baseband_samples = synthetic_channel.generate_baseband_samples()
    print(baseband_samples)
