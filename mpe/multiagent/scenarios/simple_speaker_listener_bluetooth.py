import numpy as np
from multiagent.scenarios.simple_speaker_listener import Scenario
from multiagent.scenarios.scenario_util import obscure_pos
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

COMMS_DISTANCE = 0.5
EXTRA_REWARD = False
COMM_EXISTS = False
GOAL_POS = [0, 0]

def distance(p1, p2):
    vector = p2 - p1
    # d = np.sqrt(np.sum(np.square(vector)))
    d = np.sum(np.square(vector))
    return d


class Scenario(BaseScenario):
    def make_world(self):
        print("-------")
        print(" - - - ")
        print("  ---  ")
        print("   -   ")
        print("correct")
        print("   -   ")
        print("  ---  ")
        print(" - - - ")
        print("-------")
        world = World()
        # set any world properties first
        world.dim_c = 3
        num_landmarks = 3
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.size = 0.075
            agent.communicating = False
        # speaker
        world.agents[0].movable = False
        # listener
        world.agents[1].silent = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.04
        world.stepp = 0
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        global COMM_EXISTS
        COMM_EXISTS = False
        global GOAL_POS
        GOAL_POS = [0, 0]

        # assign goals to agents
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
        # want listener to go to the goal landmark
        world.agents[0].goal_a = world.agents[1]
        world.agents[0].goal_b = np.random.choice(world.landmarks)
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])
        # random properties for landmarks
        world.landmarks[0].color = np.array([0.65, 0.15, 0.15])
        world.landmarks[1].color = np.array([0.15, 0.65, 0.15])
        world.landmarks[2].color = np.array([0.15, 0.15, 0.65])
        # special colors for goals
        world.agents[0].goal_a.color = world.agents[0].goal_b.color + np.array([0.45, 0.45, 0.45])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        return 0  # self.reward(agent, reward)

    def reward(self, agent, world):
        # squared distance from listener to landmark
        a = world.agents[0]
        dist2 = np.sum(np.square(a.goal_a.state.p_pos - a.goal_b.state.p_pos))

        # if world.stepp < 10 and EXTRA_REWARD:
        # if EXTRA_REWARD:
            # dist2 -= np.clip(0.002/dist2, -2, 0.1)
            # if a.communicating:
            #     dist2 -= 0.02
            # else:
            #     dist2 += 1
        return -dist2

    def observation(self, agent, world):
        global COMM_EXISTS
        global GOAL_POS

        if agent.name == 'agent 1':
            world.stepp = (world.stepp + 1) % 25

        goal_color = np.zeros(world.dim_color)
        goal_pos = (np.zeros(world.dim_p))

        communicating = distance(world.agents[1].state.p_pos, world.agents[0].state.p_pos) <= COMMS_DISTANCE

        if agent.goal_b is not None:
            goal_color = agent.goal_b.color

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append((entity.state.p_pos - agent.state.p_pos))

        for other in world.agents:
            if other is agent:
                continue
            entity_pos.append((other.state.p_pos - agent.state.p_pos))

        # communication of all other agents
        comm = []
        for other in world.agents:
            if other is agent or (other.state.c is None):
                continue
            if communicating:
                comm.append(other.state.c)
            else:
                comm.append(np.zeros(other.state.c.shape))

        # speaker
        if not agent.movable:
            return np.concatenate([goal_color])

        # listener
        if agent.silent:
            other = world.agents[0]

            # if communicating:
            #     goal_pos = other.goal_b.state.p_pos - agent.state.p_pos

            if communicating and not COMM_EXISTS:
                COMM_EXISTS = True
                GOAL_POS = other.goal_b.state.p_pos

            pos = GOAL_POS - agent.state.p_pos

            if COMM_EXISTS:
                comm = []
                comm.append(np.zeros(other.state.c.shape))
                entity_poss = []
                for i in range(len(entity_pos)):
                    entity_poss.append(np.zeros(2))
                return np.concatenate([pos] + entity_pos + comm)

            other_pos = other.state.p_pos - agent.state.p_pos
            return np.concatenate([pos] + entity_pos + comm)  # [other_pos]

            # return np.concatenate([agent.state.p_vel] + entity_pos + comm)  # [goal_pos] +  + [agent.state.p_vel] + entity_pos)# + comm)
            # return np.concatenate([goal_color] + [agent.state.p_vel] + entity_pos)

            # good ones
            # return np.concatenate([other_pos] + [pos] + comm)
