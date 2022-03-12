# RoboticManipulatorCore

**work in progress**
___

## Overview

Offers core functionality of a robotic manipulator with
revolute and prismatic joints.\
Provided functionalities are:
* arm matrix
* inverse arm matrix
* tool configuration vector
* first order tool configuration jacobian
* second order tool configuration jacobian (future) 
* resolved motion rate control matrix
* linear link velocities (**work in progress**)
* angular link velocities (**work in progress**)
* dexterity analysis
* gravity loading on the manipulator (**future**)
* inverse kinematics (**work in progress**)
  * Using Rapidly Exploring Random Tree
  * Bayesian Optimization Exploration
  
The system deploys torch autograd for all features related
to differentiation

The core functionality does not include any control-related
functionalites, such as constraint enforcements:\
The system
is intended to be wrapped by code adding the task specific further requirements, which might
occurr in scenarios such as control loops or reinforcement learning environments.

## Requirements

The system requires the user to be able to

* state the arm with **Denavit-Hartenberg-Parameters**
* state the center of mass of links w.r.t. **DH-frames**, if Dynamics 
  functionalities are to be used
