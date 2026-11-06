"""Per-satellite DLL/PLL tracking channel."""

import math

import numpy as np
from numpy.typing import ArrayLike

from gpsrx.ca_code import GPSPRNGenerator


class GPSChannelTracker:
    def __init__(
        self,
        prn_number: int,
        sampling_rate: float,
        initial_doppler: float,
        initial_code_phase: float,
    ):
        if not isinstance(prn_number, int) or isinstance(prn_number, bool):
            raise TypeError("prn_number must be an integer")
        if not 1 <= prn_number <= 32:
            raise ValueError("prn_number must be between 1 and 32")
        if isinstance(sampling_rate, bool) or not isinstance(
            sampling_rate, (int, float)
        ):
            raise TypeError("sampling_rate must be a real number")
        if not math.isfinite(sampling_rate) or sampling_rate <= 0.0:
            raise ValueError("sampling_rate must be finite and positive")
        if int(round(sampling_rate * 0.001)) < 1:
            raise ValueError(
                "sampling_rate must produce at least one sample per code period"
            )
        if isinstance(initial_doppler, bool) or not isinstance(
            initial_doppler, (int, float)
        ):
            raise TypeError("initial_doppler must be a real number")
        if not math.isfinite(initial_doppler):
            raise ValueError("initial_doppler must be finite")
        if isinstance(initial_code_phase, bool) or not isinstance(
            initial_code_phase, (int, float)
        ):
            raise TypeError("initial_code_phase must be a real number")
        if not math.isfinite(initial_code_phase):
            raise ValueError("initial_code_phase must be finite")

        self.prn_number = prn_number
        self.sampling_rate = float(sampling_rate)
        self.chip_rate = 1.023e6
        self.samples_per_code = int(round(self.sampling_rate * 0.001))

        # Tracking State Variables
        self.carrier_freq = initial_doppler  # Hz
        self.carrier_phase = 0.0  # radians
        self.code_phase = initial_code_phase % 1023.0  # chips (0 to 1023)

        # Loop Filter States (Simple 1st/2nd order accumulators)
        self.code_error_filtered = 0.0
        self.carrier_error_filtered = 0.0
        self.navigation_bit_accumulator = 0.0
        self.navigation_ms_count = 0
        self.last_demodulated_bit = 0

        # Correlator spacing (in chips)
        self.correlator_spacing = 0.5

        # Generate full-resolution base local PRN code (oversampled / interpolated)
        generator = GPSPRNGenerator(prn_number)
        raw_code = generator.generate_code()
        self.bipolar_code = 2.0 * np.asarray(raw_code, dtype=np.float64) - 1.0
        self.sample_indices = np.arange(self.samples_per_code, dtype=np.float64)
        self.sample_times = self.sample_indices / self.sampling_rate
        self.code_sample_offsets = (
            self.sample_indices * self.chip_rate / self.sampling_rate
        )

        self.f_l1 = 1575.42e6
        self.dt = 0.001

        # 2nd-order PLL, Bn = 18 Hz, zeta = 0.707 (error in cycles, output in Hz)
        zeta = 0.707
        wn = 18.0 * 8.0 * zeta / (4.0 * zeta**2 + 1.0)
        self.pll_tau1 = 1.0 / wn**2
        self.pll_tau2 = 2.0 * zeta / wn
        self.prev_phase_error_cycles = 0.0

        # FLL pull-in, then hand off to the PLL
        self.fll_epochs = 300
        self.fll_gain = 0.1
        self.prev_prompt: tuple[float, float] | None = None
        self.epoch = 0

        # 1st-order DLL, Bn = 2 Hz: gain = 4 * Bn * T
        self.dll_gain = 4.0 * 2.0 * self.dt

    def _generate_local_codes(
        self, current_code_phase: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Chip positions for E, P, L relative to current code phase
        # P is centered, E is shifted left (-spacing), L is shifted right (+spacing)
        p_chips = (current_code_phase + self.code_sample_offsets) % 1023
        # Early replica is advanced in code phase; late is delayed.
        e_chips = (
            current_code_phase + self.correlator_spacing + self.code_sample_offsets
        ) % 1023
        l_chips = (
            current_code_phase - self.correlator_spacing + self.code_sample_offsets
        ) % 1023

        # Nearest-neighbor or linearly interpolated code lookup
        p_code = self.bipolar_code[np.floor(p_chips).astype(np.int64) % 1023]
        e_code = self.bipolar_code[np.floor(e_chips).astype(np.int64) % 1023]
        l_code = self.bipolar_code[np.floor(l_chips).astype(np.int64) % 1023]

        return e_code, p_code, l_code

    def process_millisecond(self, signal_block: ArrayLike) -> dict[str, float | int]:
        signal = np.asarray(signal_block)
        if signal.ndim != 1:
            raise ValueError("signal_block must be one-dimensional")
        if not np.issubdtype(signal.dtype, np.number):
            raise TypeError("signal_block must contain numeric samples")
        if not np.isfinite(signal).all():
            raise ValueError("signal_block must contain only finite samples")
        if signal.size < self.samples_per_code:
            raise ValueError(
                f"Signal block must contain at least {self.samples_per_code} samples."
            )

        signal = np.asarray(signal[: self.samples_per_code], dtype=np.complex128)

        # 1. Generate Carrier Wipe Vector
        # Time vector for this 1ms block
        carrier_wipes = np.exp(
            -1j
            * (2.0 * np.pi * self.carrier_freq * self.sample_times + self.carrier_phase)
        )

        # Update carrier phase for the next millisecond boundary
        self.carrier_phase = (
            self.carrier_phase
            + 2.0
            * np.pi
            * self.carrier_freq
            * (self.samples_per_code / self.sampling_rate)
        ) % (2.0 * np.pi)

        # Apply carrier wipe to incoming samples
        wiped_signal = signal * carrier_wipes

        # 2. Generate Local Correlator Codes (Early, Prompt, Late)
        e_code, p_code, l_code = self._generate_local_codes(self.code_phase)

        # 3. Correlate Wiped Signal with E, P, L Codes
        I_e = np.dot(wiped_signal.real, e_code)
        Q_e = np.dot(wiped_signal.imag, e_code)

        I_p = np.dot(wiped_signal.real, p_code)
        Q_p = np.dot(wiped_signal.imag, p_code)

        I_l = np.dot(wiped_signal.real, l_code)
        Q_l = np.dot(wiped_signal.imag, l_code)

        # Compute amplitudes (magnitudes squared) for DLL
        E_magnitude = np.hypot(I_e, Q_e)
        L_magnitude = np.hypot(I_l, Q_l)
        P_power = I_p**2 + Q_p**2
        if not np.isfinite(E_magnitude + L_magnitude + P_power):
            raise ValueError("Correlation results must be finite")

        # 4. Discriminators (Error Calculation)
        # --- Delay Lock Loop (DLL) Normalized Amplitude Discriminator ---
        magnitude_sum = E_magnitude + L_magnitude
        code_error = (
            0.0 if magnitude_sum == 0.0 else (E_magnitude - L_magnitude) / magnitude_sum
        )

        # Costas discriminator: atan, not atan2, so data flips don't matter
        carrier_phase_error = np.arctan(Q_p / I_p) if I_p != 0.0 else 0.0
        code_error = (
            0.0 if magnitude_sum == 0.0 else (E_magnitude - L_magnitude) / magnitude_sum
        )

        # Carrier: FLL pull-in, then PLL
        if self.epoch < self.fll_epochs:
            if self.prev_prompt is not None:
                i0, q0 = self.prev_prompt
                cross = i0 * Q_p - I_p * q0
                dot = i0 * I_p + q0 * Q_p
                if dot != 0.0:
                    freq_error_hz = np.atan2(cross, dot) / (2.0 * np.pi * self.dt)
                    self.carrier_freq += self.fll_gain * freq_error_hz
            self.prev_prompt = (I_p, Q_p)
            self.prev_phase_error_cycles = carrier_phase_error / (2.0 * np.pi)
        else:
            e = carrier_phase_error / (2.0 * np.pi)
            self.carrier_freq += (self.pll_tau2 / self.pll_tau1) * (
                e - self.prev_phase_error_cycles
            ) + e * self.dt / self.pll_tau1
            self.prev_phase_error_cycles = e
        self.epoch += 1

        # Code: D = 2*delta for +/-0.5 chip E/L, so delta = D/2. Carrier-aided.
        self.code_phase += self.dll_gain * 0.5 * code_error
        self.code_phase += self.chip_rate * self.dt * self.carrier_freq / self.f_l1
        self.code_phase %= 1023.0

        # 6. Integrate the 50 Hz navigation symbol over its 20 ms duration.
        self.navigation_bit_accumulator += I_p
        self.navigation_ms_count += 1
        if self.navigation_ms_count == 20:
            self.last_demodulated_bit = (
                1 if self.navigation_bit_accumulator >= 0.0 else -1
            )
            self.navigation_bit_accumulator = 0.0
            self.navigation_ms_count = 0

        # 7. Output Metrics for Diagnostics / PVT
        return {
            "prn": self.prn_number,
            "prompt_i": I_p,
            "prompt_q": Q_p,
            "prompt_power": P_power,
            "code_phase_chips": self.code_phase,
            "doppler_hz": self.carrier_freq,
            "carrier_phase_error": carrier_phase_error,
            "code_error": code_error,
            "demodulated_bit": self.last_demodulated_bit,
        }


if __name__ == "__main__":
    from gpsrx.source import IQSource

    # channel = SyntheticGPSChannel(
    #     prn_numbers=[12],
    #     doppler_shifts=[1250.0],
    #     code_phases=[592.0],
    #     carrier_to_noise_ratios=[50.0],
    #     sampling_rate=1.023e6,
    # )

    with IQSource(4000000, "./data/captures/GPS-L1-2022-03-27.sigmf-data") as channel:
        # Initialize tracker from acquisition results
        tracker = GPSChannelTracker(
            prn_number=12,
            sampling_rate=1.023e6,
            initial_doppler=-500.0,
            initial_code_phase=592.0,
        )

        # Stream processing loop
        samples_per_ms = int(round(4000000 * 0.001))  # 1023 samples per ms at 1.023 MHz
        for ms_index in range(15000):  # Process 1 full second (2000 ms)
            signal_block = channel.get_samples(num_samples=samples_per_ms)
            if len(signal_block) < samples_per_ms:
                break

            # Run the per-millisecond tracking engine
            metrics = tracker.process_millisecond(signal_block)

            # Print telemetry every 100 ms
            if ms_index % 100 == 0:
                print(
                    f"MS {ms_index:04d}\t"
                    f"Doppler: {metrics['doppler_hz']:.1f} Hz\t"
                    f"Code Phase: "
                    f"{metrics['code_phase_chips']:.2f} chips\t"
                    f"Bit: {metrics['demodulated_bit']}"
                )
