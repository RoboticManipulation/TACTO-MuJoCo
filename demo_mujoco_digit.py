import logging
import threading
from threading import Thread
import cv2
from pathlib import Path
import hydra
import tacto.sensor_mujoco as sensor  # Import Tacto library correctly
from omegaconf import DictConfig
import dm_control.mujoco
import mujoco.viewer
from mujoco import MjModel, MjData

from dm_control import mjcf

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class SimulationRunner(Thread):
    def __init__(self, mj_model: MjModel, mj_data: MjData):
        super().__init__()
        self.mj_model = mj_model
        self.mj_data = mj_data

    def run(self) -> None:
        mujoco.viewer.launch(self.mj_model, self.mj_data)


@hydra.main(config_path="conf", config_name="digit")
def main(cfg):
    log.info("Initializing simulation environment")

    bg = cv2.imread("conf/bg_digit_240_320.jpg")

    # Initialize MuJoCo physics engine


    # Create and initialize DIGIT
    physics = dm_control.mujoco.Physics.from_xml_path(cfg.experiment.urdf_path)



    digit_body = physics.model
    digit_body_data = physics.data

    digits = sensor.Sensor(digit_body, digit_body_data, **cfg.tacto, background=bg)
    print(f"digit_body: {digit_body}")
    print(f"digit_body_data: {digit_body_data}")
    #import pdb; pdb.set_trace()

    
    digit_left_id = physics.model.body('digit_left').id
    digit_right_id = physics.model.body('digit_right').id
    #print(f"digit_left_id: {digit_left_id}")
    #import pdb; pdb.set_trace()

    digits.add_camera(1, [digit_left_id, digit_right_id])
    
    #print(digit_body.body("base_link").id)
    # add object to tacto simulator
    digits.add_body(cfg.object.urdf_path, digit_body_data.body("base_link").id, 0)

    # run p.stepSimulation in another thread
    t = SimulationRunner(mj_model=digit_body._model, mj_data=digit_body_data._data)
    t.start()

    while t.is_alive():
        color, depth = digits.render(visualize_digit=True)
        digits.updateGUI(color, depth)

    t.join()
    


if __name__ == "__main__":
    main()