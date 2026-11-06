"""GPS C/A PRN generation and oversampled code lookup."""

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
        if not isinstance(prn_number, int) or isinstance(prn_number, bool):
            raise TypeError("prn_number must be an integer")
        if prn_number not in GPS_G2_TAPS:
            raise ValueError("prn_number must be between 1 and 32")
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
