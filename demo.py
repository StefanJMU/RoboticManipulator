import numpy as np
from ManipulatorCore import Joint, ManipulatorCore

bot = ManipulatorCore([

    Joint('revolute', .5, .5, .5, .5),
    Joint('revolute', .5, .5, .5, .5),
    Joint('prismatic', 2, 3, 6, 1),
    Joint('revolute', .5, .5, .5, .5),
    Joint('revolute', .5, .5, .5, .5),
    Joint('prismatic', 2, 3, 6, 1)
])

np.set_printoptions(precision=3, suppress=True)

print(f'Arm matrix: \n {bot.arm_matrix}')
print(f'Inverse arm matrix: \n {bot.inverse_arm_matrix}')
print(f'Tool configuration vector: \n {bot.tool_config_vector}')
print(f'Tool jacobian: \n {bot.tool_jacobian}')
print(f'Resolve motion rate control matrix: \n {bot.rmrc_matrix}')
print(f'Dexterity assessment: \n {bot.degrees_of_freedom}')
