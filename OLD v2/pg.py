from engine import value
from nn import MLP, Layer, Neuron
from optim import SGD
from dt import Categorical
import random
from env import jump_env

# Step 1: Define a simple environment
class SimpleEnvironment:
    def __init__(self):
        self.state = 0
    
    def reset(self):
        self.state = value([0, 0])
        return self.state
    
    def step(self, action):
        if action == 1:
            reward = 1
            done = True
        else:
            reward = 0
            done = True
        next_state = self.state
        return next_state, reward, done

# Step 2: Define the policy network
class PolicyNetwork(MLP):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__([input_size, hidden_size, output_size], activ=['relu', ''])
    
    def forward(self, state):
        return super().__call__(state)

# Step 3: Sample actions based on the policy network's output
def select_action(policy, state):
    state_value = value(state)
    logits = policy.forward(state_value)
    dist = Categorical(logits=logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action, log_prob

# Step 4: Collect rewards and compute the policy gradient
def compute_policy_gradient(log_probs, rewards):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + 0.99 * G
        returns.insert(0, G)
    
    returns = value(returns)
    log_probs = value.combine(log_probs)
    loss = -(log_probs * returns).sum()
    loss.backward()
    return loss

# Step 5: Train the policy network
def train_policy_gradient(env, policy, optimizer, episodes=1000):
    for episode in range(episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        done = False
        
        while not done:
            action, log_prob = select_action(policy, state)
            log_probs.append(log_prob)
            next_state, reward, done = env.step(action.data)
            rewards.append(reward)
            state = next_state
        
        loss = compute_policy_gradient(log_probs, rewards)
        optimizer.step()
        policy.zero_grad()
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Loss: {loss}")


# Create the environment and policy network
env = SimpleEnvironment()
policy = PolicyNetwork(input_size=1, hidden_size=16, output_size=2)
optimizer = SGD(policy.parameters(), lr=0.01)

# Train the policy
train_policy_gradient(env, policy, optimizer, episodes=2)
