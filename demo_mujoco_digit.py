# This is a minimal example to test the digit simulator with mujoco

import logging
from pathlib import Path
from threading import Thread

import cv2
import dm_control.mujoco
import hydra
import tacto.sensor_mujoco as sensor
import mujoco.viewer
from mujoco import MjModel, MjData
log = logging.getLogger(__name__)

class SimulationRunner(Thread):
    def __init__(self, mj_model: MjModel, mj_data: MjData):
        super().__init__()
        self.mj_model = mj_model
        self.mj_data = mj_data

    def run(self) -> None:
        mujoco.viewer.launch(self.mj_model, self.mj_data)


# Load the config YAML file from examples/conf/digit.yaml
@hydra.main(config_path="conf", config_name="digit")
def main(cfg):
    # Initialize digits
    bg = cv2.imread("conf/bg_digit_240_320.jpg")


    # Initialize World
    log.info("Initializing world")


    # Create and initialize DIGIT
    physics = dm_control.mujoco.Physics.from_xml_path(cfg.experiment.urdf_path)


    digit_body = physics.model
    digit_body_data = physics.data

    digits = sensor.Sensor(digit_body, digit_body_data, **cfg.tacto, background=bg)
    digits.add_camera(1, [0])


    # add object to tacto simulator
    digits.add_body(cfg.object.urdf_path, digit_body_data.body("base_link").id, 0)

    # run p.stepSimulation in another thread
    t = SimulationRunner(mj_model=digit_body._model, mj_data=digit_body_data._data)
    t.start()

    while t.is_alive():
        color, depth = digits.render(visualize_digit=False)
        digits.updateGUI(color, depth)

    t.join()


if __name__ == "__main__":
    main()
