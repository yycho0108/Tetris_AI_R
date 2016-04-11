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
    def __init__(self, t=random(7)): #t = block type, defaults to random
        # 7 = number of possible tetris blocks
        self.t = t
        self.x=5 #middle of width
        self.y=2 #not too low
        self.r = random(4)

    def alt(self,t):
        #swap types
        self.t,t = t,self.t
        return t

    def recap(self):
        #type, rotation, x-y coord
        t = [0,0,0,0,0,0,0]
        t[self.t] = 1
        return t + [self.r, self.x,self.y]

class TetrisState:
    def __init__(self,w,h):
        self.w,self.h = w,h
        self.board = np.zeros((h,w),dtype=np.bool)
        self.block = TetrisBlock()
        self.nextBlock = random(7)
        self.act = [
                self.left,
                self.right,
                self.down,
                self.rotate,
                self.drop,
                self.alt,
                ]

        self.end = False

    def step(self,action):
        action = int(action)
        print "action : ", action
        self.act[action]()
        print "X : " , self.block.x, " Y : ", self.block.y
        #check for collision/line removal/ etc
        if self.end:
            return -1
        else:
            return 0
        #compute & return reward somehow

    def rotate(self):
        #don't want to create a wrapper method -- for performance
        self.block.r = (self.block.r+1)%4
        if self.hit():
            self.block.r = (self.block.r+3)%4 #undo

    def left(self):
        self.block.x -= 1
        if self.hit():
            self.block.x += 1 #undo

    def right(self):
        self.block.x += 1
        if self.hit():
            self.block.y -= 1 #undo

    def fillBlock(self):
        b = blocks[self.block.t][self.block.r]
        print self.block.x, self.block.y
        for pt in b:
            x = self.block.x + pt[0]
            y = self.block.y + pt[1]
            print x,y
            self.board[y][x] = True

    def down(self):
        self.block.y += 1
        if self.hit():
            self.block.y -= 1
            self.fillBlock()
            self.testLines()
            #test for line-completion

    def alt(self):
        pass
        #self.nextBlock = self.block.alt(self.nextBlock)

    def drop(self):
        while not self.hit():
            self.block.y += 1
        self.block.y -= 1
        self.fillBlock()
        self.testLines()

    def inbound(self,x,y):
        return 0<=x and x<self.w and 0<=y and y<self.h
         
    def hit(self):
        b = blocks[self.block.t][self.block.r]
        for pt in b:
            x = self.block.x + pt[0]
            y = self.block.y + pt[1]
            if (not self.inbound(x,y)) or (self.board[y][x] is True):
                return True # "WALL"
        return False

    def testLines(self): #not implemented
        print self.board.astype(np.int8)
        self.block = TetrisBlock(self.nextBlock)
        self.nextBlock = random(7)
        print "Testing Lines..."
        pass

    def recap(self):
        #print np.concatenate((self.board.flatten(),self.block.recap(),[self.nextBlock])) 
        #builds a numeric representation of itself 
        #may have to anticipate a dtype-collision
        #may have to convert?
        return np.concatenate((self.board.flatten(),self.block.recap(),[self.nextBlock])) 

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

