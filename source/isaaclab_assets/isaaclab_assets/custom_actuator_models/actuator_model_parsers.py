import os
import json
from isaaclab.actuators import ImplicitActuatorCfg

def parse_actuator_model(json_filepath: str):
    """
    This function parses an actuator model from a JSON file.
    Args:
        json_filepath: Path to the JSON file containing the actuator model.
    
    Returns:
        ImplicitActuatorCfg: Actuator model configuration.
    """

    assert os.path.exists(json_filepath), f"File not found: {json_filepath}"

    assert json_filepath.endswith(".json"), f"Invalid file extension: {json_filepath}"

    with open(json_filepath, "r") as file:
        data = json.load(file)

    return ImplicitActuatorCfg(
        joint_names_expr=data["joint_names"],
        velocity_limit=data["velocity_limit"],
        effort_limit=data["effort_limit"],
        stiffness=data["stiffness"],
        damping=data["damping"],
    )