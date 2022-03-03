from ManipulatorCore import Joint, ManipulatorCore


bot = ManipulatorCore([

    Joint('revolute',.5,.5,.5,.5),
    Joint('revolute',.5,.5,.5,.5),
    Joint('prismatic',2,3,6,1),
    Joint('revolute',.5,.5,.5,.5),
    Joint('revolute',.5,.5,.5,.5),
    Joint('prismatic',2,3,6,1) 
])

print(f'Arm matrix {bot.arm_matrix}')
print(f'Type of arm matrix : {type(bot.arm_matrix)}')
print(f'Inverse arm matrix {bot.inverse_arm_matrix}')
print(f'Tool configuration vector {bot.tool_config_vector}')
print(f'Tool jacobian {bot.tool_jacobian}')
print(f'Resolve motion rate control matrix {bot.rmrc_matrix}')
