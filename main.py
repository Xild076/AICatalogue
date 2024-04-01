import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from perlin_noise import PerlinNoise

class HelicopterEnvironment:
    def __init__(self, map_size=100, max_steps=500):
        self.map_size = map_size
        self.max_steps = max_steps
        self.current_step = 0
        self.position = None
        self.velocity = np.array([0, 0, 0])
        self.gravity = np.array([0, 0, -9.8])  # Acceleration due to gravity
        self.wind = np.random.uniform(-1, 1, size=(3,))  # Random wind
        self.goal_tolerance = 1.0
        self.grace_period = 100
        self.near_object_distance = np.random.uniform(5, 10)
        self.crashed = False

        # Generate Perlin noise map for ground
        self.noise_map = self._generate_noise_map(map_size)
        
        self.position = self._get_safe_spawn_point()

        # Generate random goal position within the map
        self.goal_pos = self._generate_goal_position()

    def reset(self):
        self.current_step = 0
        self.position = self._get_safe_spawn_point()
        self.velocity = np.array([0, 0, 0])
        self.wind = np.random.uniform(-1, 1, size=(3,))
        self.crashed = False
        return self._get_state()

    def step(self, action):
        self.current_step += 1
        reward = 0

        # Apply action to control helicopter
        lift_thrust = max(0, action[0])  # No negative thrust
        turning_thrust = action[1]
        forward_thrust = action[2]

        # Update velocity based on thrust and wind
        self.velocity += (lift_thrust * np.array([0, 0, 1]) +
                          turning_thrust * np.array([0, 1, 0]) +
                          forward_thrust * np.array([1, 0, 0])) - self.gravity + self.wind

        # Update position
        self.position += self.velocity

        # Check for collision with ground or reaching the goal
        ground_height = self._get_ground_height(self.position[0], self.position[1])
        if self.position[2] <= ground_height:
            reward = -1000 if self.current_step <= self.grace_period else -10000
            self.crashed = True
        elif np.linalg.norm(self.position - self.goal_pos) <= self.goal_tolerance:
            reward = 10000 - self.current_step
            self.goal_pos = self._generate_goal_position()  # Generate a new goal position
            self.crashed = False

        # Check if max steps reached
        done = self.current_step >= self.max_steps or self.crashed

        return self._get_state(), reward, done

    def _get_state(self):
        direction_to_goal = self.goal_pos - self.position
        distance_to_goal = np.linalg.norm(direction_to_goal)
        distance_to_ground = self.position[2] - self._get_ground_height(self.position[0], self.position[1])
        distance_to_nearest_object = self.near_object_distance
        state = np.array([
            direction_to_goal[0], direction_to_goal[1], direction_to_goal[2],
            distance_to_goal, self.wind[0], self.wind[1], self.wind[2],
            distance_to_ground, distance_to_nearest_object
        ])
        return state

    def _generate_noise_map(self, size):
        noise = PerlinNoise(octaves=6, seed=np.random.randint(1000))
        scale_factor = 0.02
        noise_map = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                noise_map[i][j] = noise([i * scale_factor, j * scale_factor])
        return noise_map

    def _get_ground_height(self, x, y):
        # Interpolate height from noise map
        x_scaled = (x + self.map_size / 2) / self.map_size * len(self.noise_map)
        y_scaled = (y + self.map_size / 2) / self.map_size * len(self.noise_map)
        x0 = int(x_scaled)
        x1 = min(x0 + 1, len(self.noise_map) - 1)
        y0 = int(y_scaled)
        y1 = min(y0 + 1, len(self.noise_map[0]) - 1)
        x_frac = x_scaled - x0
        y_frac = y_scaled - y0

        # Bi-linear interpolation
        height = (self.noise_map[x0][y0] * (1 - x_frac) * (1 - y_frac) +
                self.noise_map[x1][y0] * x_frac * (1 - y_frac) +
                self.noise_map[x0][y1] * (1 - x_frac) * y_frac +
                self.noise_map[x1][y1] * x_frac * y_frac)
        return height

    def _generate_goal_position(self):
        goal_x = np.random.uniform(-self.map_size/2, self.map_size/2)
        goal_y = np.random.uniform(-self.map_size/2, self.map_size/2)
        goal_z = self._get_ground_height(goal_x, goal_y)
        return np.array([goal_x, goal_y, goal_z])

    def _get_safe_spawn_point(self):
        while True:
            spawn_x = np.random.uniform(-self.map_size/2, self.map_size/2)
            spawn_y = np.random.uniform(-self.map_size/2, self.map_size/2)
            spawn_z = self._get_ground_height(spawn_x, spawn_y)
            if spawn_z > 0:  # Ensure spawn point is above ground
                return np.array([spawn_x, spawn_y, spawn_z])

    def get_action_space(self):
        return 3  # Lifting thrust, turning thrust, forward thrust

    def get_state_space(self):
        return 9  # Direction to goal (3), distance to goal, wind (3), distance to ground, distance to nearest object

    def render(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot ground
        X, Y = np.meshgrid(np.arange(-self.map_size/2, self.map_size/2),
                           np.arange(-self.map_size/2, self.map_size/2))
        Z = self.noise_map
        ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.8)

        # Plot helicopter
        ax.scatter(self.position[0], self.position[1], self.position[2], color='red', marker='o', s=100)

        # Plot goal
        ax.scatter(self.goal_pos[0], self.goal_pos[1], self.goal_pos[2], color='green', marker='o')

        # Plot wind direction
        ax.quiver(0, 0, 0, self.wind[0], self.wind[1], self.wind[2], color='blue', label='Wind')

        # Plot velocity direction
        ax.quiver(self.position[0], self.position[1], self.position[2],
                  self.velocity[0], self.velocity[1], self.velocity[2], color='orange', label='Velocity')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Helicopter Environment')
        ax.legend()
        plt.show()

# Example usage:
env = HelicopterEnvironment(map_size=100)
env.render()