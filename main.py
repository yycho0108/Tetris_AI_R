from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import ActionValueNetwork
from pybrain.rl.learners.valuebased import Q, NFQ, SARSA

from pybrain.rl.explorers import EpsilonGreedyExplorer
from pybrain.rl.environments.environment import Environment
from pybrain.rl.environments.episodic import EpisodicTask, EpisodicExperiment

import numpy as np

def random(n):
    return np.random.randint(n)


blocks = [
     [ [0,0,1,0,2,0,-1,0], [0,0,0,1,0,-1,0,-2], [0,0,1,0,2,0,-1,0], [0,0,0,1,0,-1,0,-2] ],
     [ [0,0,1,0,0,1,1,1], [0,0,1,0,0,1,1,1], [0,0,1,0,0,1,1,1], [0,0,1,0,0,1,1,1] ],
     [ [0,0,-1,0,0,-1,1,-1], [0,0,0,1,-1,0,-1,-1], [0,0,-1,0,0,-1,1,-1], [0,0,0,1,-1,0,-1,-1] ],
     [ [0,0,-1,-1,0,-1,1,0], [0,0,-1,0,-1,1,0,-1], [0,0,-1,-1,0,-1,1,0], [0,0,-1,0,-1,1,0,-1] ],
     [ [0,0,-1,0,1,0,-1,-1], [0,0,0,-1,0,1,-1,1], [0,0,-1,0,1,0,1,1], [0,0,0,-1,0,1,1,-1] ],
     [ [0,0,1,0,-1,0,1,-1], [0,0,0,1,0,-1,-1,-1], [0,0,1,0,-1,0,-1,1], [0,0,0,-1,0,1,1,1] ],
     [ [0,0,-1,0,1,0,0,1], [0,0,0,-1,0,1,1,0], [0,0,-1,0,1,0,0,-1], [0,0,-1,0,0,-1,0,1] ],
]
blocks = np.reshape(blocks,(7,4,4,2)) #type, rotation, 4 [points(x,y)]

class TetrisBlock:
    #more like a container
    def __init__(self): #t = block type, defaults to random
        # 7 = number of possible tetris blocks
        self.t = random(7)
        self.x=0
        self.y=0
        self.r=0

    def alt(self,t):
        #swap types
        self.t,t = t,self.t
        return t
    def recap(self):
        #type, rotation, x-y coord
        return [self.t,self.r,self.x,self.y]

class TetrisState:
    def __init__(self,w,h):
        self.w,self.h = w,h
        self.board = np.zeros((h,w),dtype=np.int8)
        self.block = TetrisBlock()
        self.altBlock = random(7)
        self.acts = [
                self.left,
                self.right,
                self.down,
                self.rotate,
                self.drop,
                self.alt,
                ]

        self.end = False

    def step(action):
        self.acts[action]()
        val = self.hit()
        if val == -1: #wall
        #check for collision/line removal/ etc
        if self.end:
            return -1
        else:
            return 0
        #compute & return reward somehow

    def rotate(self):
        #don't want to create a wrapper method -- for performance
        self.block.r = (self.block.r+1)%4

    def left(self):
        self.block.x -=1
        if self.hit():
            self.block.x += 1 #undo

    def right(self):
        self.block.x += 1
        if self.hit():
            self.block.y -= 1

    def down(self):
        self.block.y += 1

    def alt(self):
        self.altBlock = self.block.alt(self.altBlock)

    def drop(self):
        while self.hit() == 0:
            self.block.y += 1

    def hitside(self,x,y):
        return x<0 or x >= self.w

    def hitbottom(self,x,y):
        return self.board[x][y] ==1 or y>=self.h
         

    def hit(self):
        b = block[self.block.t][self.block.r]
        for pt in b:
            x = self.block.x + pt[0]
            y = self.block.y + pt[1]
            if not inbound(x,y):
                return -1 # "WALL"
            elif hitbottom(x,y):
                return 1 #
        return 0 #empty

    def recap(self):
        #builds a numeric representation of itself 
        #may have to anticipate a dtype-collision
        #may have to convert?
        return np.concatenate((self.board.flatten(),self.block.recap(),[self.altBlock])) 

class TetrisEnv(Environment):
    def __init__(self,w,h):
        self.w = w
        self.h = h
        self.indim = 6
        self.outdim = w*h+11
        self.state = TetrisState(w,h)

    def getSensors(self):
        return self.state.recap()

    def performAction(self,action):
        #return reward
        return self.state.step(action)

    def reset(self):
        self.state = TetrisState(self.w, self.h)
        pass

    def end(self):
        return self.state.end

class TetrisTask(EpisodicTask):
    def __init__(self,env):
        self.env = env
        self.reward=0
        self.discount = 0.99 #gamma
    #def addReward():
    #    pass
    #    seems to be implemented already?
    def getTotalReward():
        # total score?
        pass

    def isFinished(self):
        return self.env.end()

    def performAction(self,action):
        self.reward = self.env.performAction(action)
        #observe reward...

    def getObservation(self):
        return self.env.getSensors()

    def getReward(self):
        return self.reward

    def reset(self):
        #i suppose this is the proper way to do it?
        EpisodicTask.reset(self)
        self.env.reset()

    @property
    def indim(self):
        return self.env.indim

    @property
    def outdim(self):
        return self.env.outdim



env = TetrisEnv(10,20) #Tetris
task = TetrisTask(env)

QNet = ActionValueNetwork(10*20+11, 6);

learner = NFQ(); #Q()?
learner._setExplorer(EpsilonGreedyExplorer(0.2,decay=0.99))

agent = LearningAgent(QNet,learner);

experiment = EpisodicExperiment(task,agent)

while True:
    experiment.doEpisodes(1)
    agent.learn()
    agent.reset() #or call more sporadically...?
    task.reset()

