import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 9
        self.action_low = 100
        self.action_high = 2000
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #reward = 1.-.3*sigmoid((abs(self.sim.pose[:3] - self.target_pos)).sum())
        #reward -= .3*sigmoid((abs(self.sim.pose[-3:] - np.array([0., 0., 0.]))).sum())
        #reward = 1.-.3*sigmoid(abs(self.sim.pose - np.concatenate((self.target_pos, np.array([0., 0., 0.])), axis=None)).sum())
        #reward = 1 - np.tanh(abs(self.target_pos[:2] - self.sim.pose[:2]).sum()/4 + abs(self.target_pos[2] - self.sim.pose[2])/2 + abs(self.sim.pose[-3:] - np.array([0., 0., 0.])).sum()/4 )
        #+ abs(self.sim.v[2]/8)
        dist = np.sqrt(np.sum(np.square(self.sim.pose[:3] - self.target_pos)))/30
        dist_discount = 1 - np.power(dist, 0.8) # Pythagorean theorem, scaled between 0 and 1
        z_vel_discount = 1 - abs(np.power(max(self.sim.v[2], 0.1), (1/dist)))
        reward = dist_discount * z_vel_discount
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        v_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
            v_all.append(self.sim.v)
        next_state = np.concatenate((pose_all, v_all), axis=None)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        single_snap = np.concatenate((self.sim.pose, self.sim.v), axis=None)
        state = np.concatenate([single_snap] * self.action_repeat) 
        return state

def sigmoid(x):
    return 1 / (1 + np.exp(-x))