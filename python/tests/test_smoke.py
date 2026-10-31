from gpsrx.config import ReceiverConfig


def test_default_configuration() -> None:
    config = ReceiverConfig()
    assert config.sample_rate == 2_046_000.0
    assert config.integration_ms == 1
