from dataclasses import dataclass

@dataclass
class BridgeConfig:
    address: str = "*"
    state_port: int = 1801
    command_port: int = 1802
    force_port: int = 1803
    force_sensor_name: str = None


@dataclass
class SimulationParameters:
    device: str = "cuda:0"
    dt: float = 1/60
