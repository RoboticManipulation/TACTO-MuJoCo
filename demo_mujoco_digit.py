import logging
from threading import Thread
import cv2
from pathlib import Path
import hydra
import tacto.sensor_mujoco as sensor  # Import Tacto library correctly
from omegaconf import DictConfig
import dm_control.mujoco
import mujoco.viewer
from mujoco import MjModel, MjData

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


def update_tacto_simulation(digit_sensor):
    # Function to update Tacto simulation
    while True:  # Add your own condition to stop updating
        color, depth = digit_sensor.render(visualize_digit=False)
        digit_sensor.update(color, depth)

@hydra.main(config_path="conf", config_name="digit")
def main(cfg):
    log.info("Initializing simulation environment")

    bg = cv2.imread("conf/bg_digit_240_320.jpg")

    # Initialize MuJoCo physics engine
    physics = dm_control.mujoco.Physics.from_xml_path(cfg.experiment.urdf_path)

    # Initialize DIGIT sensor
    digit_left_model = physics.model.body("digit_left")
    digit_left_data = physics.data.body("digit_left").id
    digit_sensor = sensor.Sensor(digit_left_model, digit_left_data, **cfg.tacto, background=bg)
    digit_sensor.add_camera(1, [0])

    # Start the simulation runner
    simulation_thread = SimulationRunner(physics.model._model, physics.data._data)
    simulation_thread.start()


    # Start Tacto simulation update thread
    tacto_thread = Thread(target=update_tacto_simulation, args=(digit_sensor))
    tacto_thread.start()

    # Wait for threads to complete
    simulation_thread.join()
    tacto_thread.join()

if __name__ == "__main__":
    main()