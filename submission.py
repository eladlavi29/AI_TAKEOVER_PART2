from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random


# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    h = 0
    robot = env.get_robot(robot_id)

    if env.robot_is_occupied(robot_id):
        doesHoldBox = 1
    else:
        doesHoldBox = 0

    h += robot.battery + robot.credit + 2 * doesHoldBox

    max = -100

    for package in env.packages:
        if doesHoldBox == 1:
            cost = manhattan_distance(robot.position, robot.package.destination) + 1
            utility = manhattan_distance(robot.package.position, robot.package.destination) * 2 - cost

        else:
            cost = manhattan_distance(robot.position, package.position) + manhattan_distance(package.destination, package.position) + 2
            utility = manhattan_distance(package.position, package.destination) * 2 - cost

        if(utility > max):
            max = utility

    if(max == -100):
        min = manhattan_distance(robot.position, env.charge_stations[0].position)

        if min > manhattan_distance(robot.position, env.charge_stations[1].position):
            min = manhattan_distance(robot.position, env.charge_stations[1].position)

        max = 10 - min

        print(min)

    return h + max
class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move south", "move north", "move south", "move north",
                           "move south", "move north", "move south", "move north", "move south"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)