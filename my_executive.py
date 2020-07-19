# name : yuval saadati
# id: 205956634
import sys

from pddlsim.local_simulator import LocalSimulator
from my_agent import Executor
print LocalSimulator().run(str(sys.argv[2]), str(sys.argv[3]), Executor(str(sys.argv[1]), str(sys.argv[4])))