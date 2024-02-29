from dm_control import mjcf

class Model(object):
  def __init__(self, filename):
    with open(filename) as f:
        self.mjcf_model = mjcf.from_file(f)

    # self.mjcf_model = mjcf.RootElement(model=name)

    # self.upper_arm = self.mjcf_model.worldbody.add('body', name='upper_arm')
    # self.shoulder = self.upper_arm.add('joint', name='shoulder', type='ball')
    # self.upper_arm.add('geom', name='upper_arm', type='capsule',
    #                    pos=[0, 0, -0.15], size=[0.045, 0.15])

    # self.forearm = self.upper_arm.add('body', name='forearm', pos=[0, 0, -0.3])
    # self.elbow = self.forearm.add('joint', name='elbow',
    #                               type='hinge', axis=[0, 1, 0])
    # self.forearm.add('geom', name='forearm', type='capsule',
    #                  pos=[0, 0, -0.15], size=[0.045, 0.15])


def main():
    # Parse from path
    mjcf_model_arm = Model(filename='sawyer.xml')

    mjcf_model_gripper = Model(filename='gripper.xml')
    mjcf_model_digit_left = Model(filename='digit.xml')
    mjcf_model_digit_right = Model(filename='digit.xml')


    mjcf_model_arm.worldbody.find('body', 'foo')
    print(type(mjcf_model))  # <type 'mjcf.RootElement'>



    body = UpperBody()
    physics = mjcf.Physics.from_mjcf_model(body.mjcf_model)


if __name__ == "__main__":
    main()