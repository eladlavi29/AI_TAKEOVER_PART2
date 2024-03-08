from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import numpy as np
import time

# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    #TODO: print heuristic of mission and h at different points


    h = 0
    robot = env.get_robot(robot_id)

    if env.robot_is_occupied(robot_id):
        doesHoldBox = 1
    else:
        doesHoldBox = 0

    if robot.credit > 15:
        h += 10 * robot.credit + doesHoldBox
    else:
        h += 4 * robot.battery + 3 * robot.credit + doesHoldBox

    package = env.packages[0]
    if doesHoldBox == 1:
        cost = manhattan_distance(robot.position, robot.package.destination) + 1
        mission = manhattan_distance(robot.package.position, robot.package.destination) * 2 - cost

    else:
        cost = manhattan_distance(robot.position, package.position) + manhattan_distance(package.destination, package.position) + 2
        mission = manhattan_distance(package.position, package.destination) * 2 - cost

    if(cost > robot.battery):
        min = manhattan_distance(robot.position, env.charge_stations[0].position)

        if min > manhattan_distance(robot.position, env.charge_stations[1].position):
            min = manhattan_distance(robot.position, env.charge_stations[1].position)

        mission = 10 - min

    max_cost = 1000

    return h + mission + max_cost
class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):

    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)
    # TODO: section b : 1
    def RBminimax(self, env: WarehouseEnv, agent_id, D, turn=True):
        if env.done() or D == 0:
            return self.heuristic(env, agent_id), self.successors(env,agent_id)[0][0]
        chosen_step = None
        if turn:
            curmax = -np.inf
            operators, children = self.successors(env, agent_id)
            for operator, child in zip(operators, children):
                v = self.RBminimax(child, agent_id, D-1, not turn)[0]

                if(v > curmax):
                    curmax = v
                    chosen_step = operator

            return curmax, chosen_step

        curmin = np.inf
        operators, children = self.successors(env,1-agent_id)
        for operator, child in zip(operators, children):
            v = self.RBminimax(child, agent_id, D-1, not turn)[0]

            if (v < curmin):
                curmin = v
                chosen_step = operator
        return curmin, None

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        time_factor = 5
        previous_iteration_duration = 0
        d = 1

        step = None

        while True:
            if time_limit < time_factor * previous_iteration_duration:
                return step

            d += 1
            start_time = time.time()
            step = self.RBminimax(env, agent_id, D=d, turn=True)[1]
            previous_iteration_duration = time.time() - start_time
            time_limit -= previous_iteration_duration

class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)

    def ABminimax(self, env: WarehouseEnv, agent_id, D, turn=True,a=-np.inf,b=np.inf):
        if env.done() or D == 0:
            return self.heuristic(env, agent_id), self.successors(env,agent_id)[0][0]
        chosen_step = None

        if turn:
            curmax = -np.inf
            operators, children = self.successors(env, agent_id)
            for operator, child in zip(operators, children):
                v = self.ABminimax(child, agent_id, D - 1, not turn,a,b)[0]
                if (v > curmax):
                    curmax = v
                    chosen_step = operator
                a = max(curmax,a)
                if curmax >= b:
                    return np.inf,None
            return curmax, chosen_step

        curmin = np.inf
        operators, children = self.successors(env, 1-agent_id)
        for operator, child in zip(operators, children):
            v = self.ABminimax(child, agent_id, D - 1, not turn,a,b)[0]
            if (v < curmin):
                curmin = v
                chosen_step = operator
            b = min(curmin, b)
            if a >= curmin:
                return -np.inf, None
        return curmin, None

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        time_factor = 20
        previous_iteration_duration = 0
        d = 1
        step = None
        while True:
            if time_limit < time_factor * previous_iteration_duration:
                return step
            d += 1
            start_time = time.time()
            step = self.ABminimax(env, agent_id, D=d, turn=True)[1]
            previous_iteration_duration = time.time() - start_time
            time_limit -= previous_iteration_duration

class AgentExpectimax(Agent):
    # TODO: section d : 1
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)
    #
    def probabilities(self,env: WarehouseEnv, agent_id,operator):
        legal_ops = self.successors(env,agent_id)[0]
        if operator not in legal_ops:
            return 0.0
        is_move_right = ('move east' in legal_ops)
        is_collect_package = ('pick up' in legal_ops)
        sum = len(legal_ops) + is_move_right + is_collect_package
        if operator == 'move east' or operator == 'pick up':
            return 2/sum
        else:
            return 1/sum

    def expectimax(self, env: WarehouseEnv, agent_id, D, turn=True):
        if env.done() or D == 0:
            return self.heuristic(env, agent_id), self.successors(env,agent_id)[0][0]
        chosen_step = None
        if turn:
            curmax = -np.inf
            operators, children = self.successors(env, agent_id)
            for operator, child in zip(operators, children):
                v = self.expectimax(child, agent_id, D-1, not turn)[0]
                if(v > curmax):
                    curmax = v
                    chosen_step = operator
            return curmax, chosen_step
        expectation = 0
        operators, children = self.successors(env, 1-agent_id)
        for operator, child in zip(operators, children):
            v = self.expectimax(child, agent_id, D-1, not turn)[0]
            expectation += self.probabilities(env,agent_id,operator)*v
        return expectation, None

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):

        time_factor = 10
        previous_iteration_duration = 0
        d = 1
        step = None
        while True:
            if time_limit < time_factor * previous_iteration_duration:
                print(d)
                return step
            d += 1
            start_time = time.time()
            step = self.expectimax(env, agent_id, D=d, turn=True)[1]
            previous_iteration_duration = time.time() - start_time
            time_limit -= previous_iteration_duration



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