import numpy as np
from ManipulatorCore import Joint, ManipulatorCore
import torch

ph = np.pi / 2
bot = ManipulatorCore([

    Joint('revolute', 0, 450, 0, ph),
    Joint('prismatic', 0, 10, 0, ph),
    Joint('revolute', 0, 250, 0, 0)

])

np.set_printoptions(precision=3, suppress=True)

#print(f'Arm matrix: \n {bot.arm_matrix}')
#print(f'Inverse arm matrix: \n {bot.inverse_arm_matrix}')
#print(f'Inverse arm matrix: \n {bot.inverse_arm_matrix}')
#print(f'Tool configuration vector: \n {bot.tool_config_vector}')
#print(f'Tool jacobian: \n {bot.tool_jacobian}')
#print(f'Resolve motion rate control matrix: \n {bot.rmrc_matrix}')
#print(f'Dexterity assessment: \n {bot.degrees_of_freedom}')

dist, config = bot.inverse_kinematics(np.array([200, -200, 200, 0, 0, -1]),
                                      [(-4, 4), (0, 700), (-4, 4)],
                                      np.eye(3),
                                      np.array([-1000, -1000, -1000]),
                                      np.array([1000, 1000, 1000]))
print(f'Config {config} yielding a score of {dist}')

bot.update_config(config)
print(bot.tool_config_vector)
