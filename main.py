from Environment import Environment

env = Environment("assets/environment.xml", timeStep = 0.001)
env.compile()
env.runSimulation(
    showLeftUI  = False,
    showRightUI = False,
)