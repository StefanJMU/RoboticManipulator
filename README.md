# RoboticManipulatorCore
___

## Overview

Offers core functionality of a robotic manipulator with
revolute and prismatic joints.\
Provided functionalities are:
* arm matrix
* inverse arm matrix
* tool configuration vector
* tool configuration jacobian (first oder)
* tool configuration jacobian (second order)
* resolved motion rate control matrix
* linear link velocities
* angular link velocities
* dexterity analysis
* gravity loading on the manipulator (**future**)
* inverse kinematics (**future**)

The core functionality does not include any control-related
functionalites, such as constraint enforcements:\
The system
is intended to be wrapped by code adding the task specific further requirements, which might
occurr in scenarios such as control loops or reinforcement learning environments.

## Requirements

The system requires the user to be able to

* state the arm with **Denavit-Hartenberg-Parameters**
* state the center of mass of links w.r.t. **DH-frames**, if Dynamics
* functionalities are to be used
