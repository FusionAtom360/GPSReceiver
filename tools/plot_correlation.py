"""Plot acquisition correlation surfaces for the technical report."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from gpsrx.acquisition import FFTAcquisition


def plot_correlation_surface(
    acq: FFTAcquisition, signal_block: ArrayLike, doppler_bins: Sequence[float]
) -> None:
    """Compute and display the Doppler/code-phase correlation surface."""
    signal = np.asarray(signal_block)
    if signal.ndim != 1:
        raise ValueError("signal_block must be one-dimensional")
    if not np.issubdtype(signal.dtype, np.number):
        raise TypeError("signal_block must contain numeric samples")
    if not np.isfinite(signal).all():
        raise ValueError("signal_block must contain only finite samples")

    bins = np.asarray(list(doppler_bins), dtype=np.float64)
    if bins.ndim != 1 or bins.size == 0:
        raise ValueError(
            "doppler_bins must be a non-empty one-dimensional sequence"
        )
    if not np.isfinite(bins).all():
        raise ValueError("doppler_bins must contain only finite values")

    required_samples = acq.samples_per_code * acq.noncoherent_ms
    if signal.size < required_samples:
        raise ValueError(
            f"Signal block length must be at least {required_samples} samples "
            f"({acq.noncoherent_ms} ms)."
        )

    signal = np.asarray(signal[:required_samples], dtype=np.complex128)
    sample_numbers = np.arange(acq.samples_per_code, dtype=np.float64)
    segment_scores = np.zeros(
        (bins.size, acq.samples_per_code), dtype=np.float64
    )

    for segment in range(acq.noncoherent_ms):
        start = segment * acq.samples_per_code
        stop = start + acq.samples_per_code
        segment_signal = signal[start:stop]
        segment_time = (sample_numbers + start) / acq.sampling_rate
        carrier_wipe = np.exp(-1j * 2.0 * np.pi * bins[:, None] * segment_time)
        correlations = np.fft.ifft(
            np.fft.fft(segment_signal[None, :] * carrier_wipe, axis=1)
            * acq.code_fft_conj[None, :],
            axis=1,
        )
        segment_scores += np.abs(correlations) ** 2

    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.imshow(
        segment_scores,
        aspect="auto",
        extent=(0.0, float(acq.samples_per_code), bins[-1], bins[0]),
        cmap="viridis",
    )
    ax.set_title(f"Acquisition Correlation Surface (PRN {acq.prn_number})")
    ax.set_xlabel("Code Phase (chips)")
    ax.set_ylabel("Doppler Bin (Hz)")
    fig.colorbar(cax, label="Squared Magnitude")
    plt.tight_layout()
    plt.show()


def main() -> int:
    from gpsrx.synthetic import SyntheticGPSChannel

    channel = SyntheticGPSChannel(
        prn_numbers=[12],
        doppler_shifts=[1250.0],
        code_phases=[450.2],
        carrier_to_noise_ratios=[42.0],
    )
    signal_block = channel.generate_baseband_samples(num_samples=1023)
    acq = FFTAcquisition(prn_number=12)
    doppler_grid = np.arange(-5000.0, 5001.0, 500.0).tolist()
    plot_correlation_surface(acq, signal_block, doppler_grid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
