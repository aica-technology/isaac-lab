from dataclasses import dataclass


@dataclass
class BridgeConfig:
    address: str = "*"
    state_port: int = 1801
    command_port: int = 1802
    ft_sensor_port: int = 1803
    ft_sensor_name: str = None
