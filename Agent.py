import torch
import mujoco

class Agent():
    def __init__(self, path:str, jointNames:list, actuatorNames:list, sensorNames:list):
        self._mjSpec        = mujoco.MjSpec.from_file(path)
        self._jointNames    = jointNames
        self._actuatorNames = actuatorNames
        self._sensorNames   = sensorNames

        self._mjModel = None
        self._mjData  = None

    def bind(self, model, data):
        self._mjModel = model
        self._mjData = data

    def getSpec(self):
        return self._mjSpec


