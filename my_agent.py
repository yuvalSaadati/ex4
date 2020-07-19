# name : yuval saadati
# id: 205956634
import pddlsim
import random
import json

from valid_actions import ValidActions
from valid_actions import PythonValidActions
# save the last action
last_return_action = ""
# save the last state
last_state = None
# save the Q-table in dictionary
q_table = None
# save the best policy in dictionary
policy_table = {}


class Executor(object):
    def __init__(self, arg, policy_file):
        super(Executor, self).__init__()
        # -L is for learning and -E is for execute the optimal policy
        self.arg = arg
        # policy file name
        self.policy_file = policy_file

    def initialize(self, services):
        self.services = services
        self.valid_actions_options = ValidActions(self.services.parser, self.services.pddl, self.services.perception)
        self.python_valid_actions_options = PythonValidActions(self.services.parser, self.services.perception)

    def build_Q_table(self):
        # build in the first learning Q-table
        table = { "c00": {},
                 "c0": {},
                 "c1": {},
                 "c2": {},
                 "c3": {},
                 "g0": {},
                 "g1": {},
                 "g2": {},
                 "g3": {},
                 "g4": {},
                 "d0": {},
                 "d1": {},
                 "d2": {},
                 "d3": {},
                 "d4": {},
                 "epsilon" : 0.1,
                 "alpha" : 0.1,
                 "gamma" : 0.6}
        return table

    def reward_function(self, action):
        global last_return_action
        if "pick-food" in action:
            return 20
        elif self.get_stuck(action):
            # got to dead end
            return -10
        else:
            # just a step
            return -1

    def get_stuck(self, action):
        # the agent reached to dead end
        action_words = action.split()
        if len(action_words) == 4:
            if action_words[3] == "g4" or action_words[3] == "d4" or action_words[3] == "c00":
                return True
        return False

    def action_made_impact(self):
        # check if the last action did not take place
        global last_return_action
        state = self.services.perception.get_state()
        need_to_be_in = last_return_action.split()
        size = len(need_to_be_in)
        # check if the agent is where it is supposed to be
        for at_now in state['at']:
            if size == 4:
                if need_to_be_in[3] == at_now[1]:
                    return True
            elif size == 4:
                if need_to_be_in[3] == at_now[1] or need_to_be_in[4] == at_now[1]:
                    return True
        return False

    def get_at_tile(self, state):
        # return the agent location
        for at_now in state['at']:
            return at_now[1]

    def max_action_Qvalue(self, state):
        # return the highest value in a particular state from Q-table
        global q_table
        actions = self.valid_actions_options.get(state)
        max_value = 0
        flag = True
        actions_dic = q_table[self.get_at_tile(state)]
        for action in actions:
            for key, value in actions_dic.iteritems():
                if flag:
                    flag = False
                    max_value = value
                if value >= max_value:
                    if key == action:
                        max_value = value
        return max_value

    def build_table(self):
        # initialize the table in zeros
        global q_table
        state = self.services.perception.get_state()
        actions = self.valid_actions_options.get(state)
        actions_dic = q_table[self.get_at_tile(state)]
        for action in actions:
            if action not in actions_dic.keys():
                q_table[self.get_at_tile(state)][action] = 0

    def max_action_Qname(self, state):
        # returns the action name with the highest value
        global q_table
        actions = self.valid_actions_options.get(state)
        action_name = ""
        max_value = 0
        flag = True
        actions_dic = q_table[self.get_at_tile(state)]
        for key, value in actions_dic.iteritems():
            if flag:
                flag = False
                max_value = value
                action_name = random.choice(actions)
            if value >= max_value:
                if key in actions:
                    max_value = value
                    action_name = key
        return action_name

    def best_policy(self):
        # create optimal policy and put it in a file
        global policy_table
        global q_table
        max = 0
        max_action_name = ""
        for state, actions_dic in q_table.iteritems():
            first = True
            if state == "epsilon" or state =="alpha" or state == "gamma":
                continue
            for action_name, action_value in actions_dic.iteritems():
                if first:
                    max = action_value
                    max_action_name = action_name
                    first = False
                if action_value > max:
                    max = action_value
                    max_action_name = action_name
            policy_table[state] = max_action_name

        jsonFile = open(self.policy_file, "w+")
        jsonFile.write(json.dumps(policy_table))
        jsonFile.close()

    def next_action(self):
        # Return the next action to apply
        global last_return_action
        global last_state
        global q_table
        global policy_table
        if self.arg == "-L":
            # start to learn
            if self.services.goal_tracking.reached_all_goals():
                # the agent reached thr goal so q-table and optimal policy file will be update
                jsonFile = open("q_table.json", "w+")
                jsonFile.write(json.dumps(q_table))
                jsonFile.close()
                self.best_policy()
                return None

            state = self.services.perception.get_state()
            actions = self.valid_actions_options.get(state)
            first = False
            if last_state is None:
                try:
                    # this is the first iteration of the agent but the Q-table created before
                    with open("q_table.json", 'rb') as file:
                        q_table = json.load(file)
                        self.build_table()
                        random_action = random.choice(actions)
                        last_return_action = random_action
                        last_state = state
                        return random_action
                except IOError:
                    # the is the first iteration of learning
                    first = True
                if first:
                    # build for the first time Q-table
                    q_table = self.build_Q_table()
                    self.build_table()
                    random_action = random.choice(actions)
                    actions_dic = q_table[self.get_at_tile(state)]
                    if random_action not in actions_dic.keys():
                        q_table[self.get_at_tile(state)][random_action] = 0
                    # update Q-table
                    with open("q_table.json", 'w+') as file3:
                        file3.write(json.dumps(q_table))
                    last_return_action = random_action
                    last_state = state
                    return random_action

            else:
                # not the first iteration so Q-table exists
                with open('q_table.json') as f:
                    q_table = json.load(f)
                    self.build_table()
                f.close()
            if random.uniform(0, 1) < q_table["epsilon"]:
                # choose random action
                random_action = random.choice(actions)
                # get reward by the random action
                reward = self.reward_function(random_action)
                old_value = q_table[self.get_at_tile(last_state)][last_return_action]
                next_max = self.max_action_Qvalue(state)
                alpha = q_table["alpha"]
                gamma = q_table["gamma"]
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                q_table[self.get_at_tile(state)][random_action] = new_value
                if not self.action_made_impact():
                    q_table[self.get_at_tile(last_state)][last_return_action] += alpha * -0.5
                # save the last state
                last_state = state
                # save the last action
                last_return_action = random_action
                jsonFile = open("q_table.json", "w+")
                jsonFile.write(json.dumps(q_table))
                jsonFile.close()
                return random_action
            else:
                # get the action with the highest value
                action = self.max_action_Qname(state)
                reward = self.reward_function(action)
                old_value = q_table[self.get_at_tile(last_state)][last_return_action]
                next_max = self.max_action_Qvalue(state)
                alpha = q_table["alpha"]
                gamma = q_table["gamma"]
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                q_table[self.get_at_tile(state)][action] = new_value
                if not self.action_made_impact():
                    q_table[self.get_at_tile(last_state)][last_return_action] += alpha * -0.5
                jsonFile = open("q_table.json", "w+")
                jsonFile.write(json.dumps(q_table))
                jsonFile.close()
                last_state = state
                last_return_action = action
                return action
        elif self.arg == "-E":
            # execute the optimal policy
            with open(self.policy_file, 'rb') as file:
                # get the optimal policy from file
                policy_table = json.load(file)
            if self.services.goal_tracking.reached_all_goals():
                return None
            state = self.services.perception.get_state()
            return policy_table[self.get_at_tile(state)]
        else:
            if self.services.goal_tracking.reached_all_goals():
                return None
            state = self.services.perception.get_state()
            actions = self.valid_actions_options.get(state)
            return random.choice(actions)