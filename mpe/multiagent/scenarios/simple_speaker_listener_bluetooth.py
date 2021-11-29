import numpy as np
from multiagent.scenarios.simple_speaker_listener import Scenario
from multiagent.scenarios.scenario_util import obscure_pos
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
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
        # assign goals to agents
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
        # want listener to go to the goal landmark
        world.agents[0].goal_a = world.agents[1]
        world.agents[0].goal_b = np.random.choice(world.landmarks)
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25,0.25,0.25])
        # random properties for landmarks
        world.landmarks[0].color = np.array([0.65,0.15,0.15])
        world.landmarks[1].color = np.array([0.15,0.65,0.15])
        world.landmarks[2].color = np.array([0.15,0.15,0.65])
        # special colors for goals
        world.agents[0].goal_a.color = world.agents[0].goal_b.color + np.array([0.45, 0.45, 0.45])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        return 0 #self.reward(agent, reward)

    def reward(self, agent, world):
        # squared distance from listener to landmark
        a = world.agents[0]
        dist2 = np.sum(np.square(a.goal_a.state.p_pos - a.goal_b.state.p_pos))
        # print(world.stepp)
        if world.stepp < 0:
            if a.communicating:
                # print("-")
                dist2 -= 1
            else:
                # print("exxx")
                dist2 += 1
        return -dist2

    def observation(self, agent, world):
        # goal color\

        if agent.name == 'agent 1':
            world.stepp = (world.stepp + 1) % 25
        # print("aaa")
        goal_color = np.zeros(world.dim_color)
        if agent.goal_b is not None:
            goal_color = agent.goal_b.color

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            # entity_pos.append(obscure_pos(agent.state.p_pos, entity.state.p_pos))
            entity_pos.append((entity.state.p_pos - agent.state.p_pos))

        # communication of all other agents
        comm = []
        for other in world.agents:
            if other is agent or (other.state.c is None):
                continue
            if np.sum(obscure_pos(other.state.p_pos, agent.state.p_pos)) > 0:
                comm.append(other.state.c)
                agent.communicating = True
            else:
                comm.append(np.zeros(other.state.c.shape))
                agent.communicating = False

        # speaker
        if not agent.movable:
            return np.concatenate([goal_color])
        # listener
        if agent.silent:
            for other in world.agents:
                if other is agent or (other.state.c is None):
                    continue
                if agent.communicating > 0:
                    goal_color = other.goal_b.color
                else:
                    goal_color = (np.zeros(world.dim_color))

            return np.concatenate([goal_color] + [agent.state.p_vel] + entity_pos + comm)
            # return np.concatenate([goal_color] + [agent.state.p_vel] + entity_pos)

