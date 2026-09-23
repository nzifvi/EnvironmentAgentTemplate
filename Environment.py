import mujoco
import mujoco.viewer

class Environment:
    def __init__(self, path:str, timeStep:float):
        self._mjSpec  = mujoco.MjSpec(path)
        self._mjModel = None
        self._mjData  = None

        self._timeStep = timeStep
        self._agents   = []

    def addAgent(self, agent, pos) -> None:
        self._mjSpec.attach(
            agent,
            prefix = f"agent{len(self._agents)}",
            pos = list(pos)
        )
        (self._agents.append(agent))

    def compile(self) -> None:
        self._mjModel = self._mjSpec.compile()
        self._mjData  = mujoco.MjData(self._mjModel)
        self._mjModel.opt.timestep = self._timeStep

        for agent in self._agents:
            agent.bind(
                model = self._mjModel,
                data  = self._mjData
            )

    def runSimulation(self, simFunction = None) -> None:
        viewer = mujoco.viewer.launch_passive(
            self._mjModel,
            self._mjData,
        )
        while viewer.is_running():
            if simFunction is not None:
                simFunction()

            mujoco.mj_step(
                self._mjModel,
                self._mjData,
            )
