#Tetris AI with PyBrain

PyBrain offers a rich set of tools for machine-learning; I will implement a custom Tetris Environment, as well as tasks and agents, and observe its progress.

---
## TO DO:

- Implement Game Logic
	- [x] Collision Checking
	- [x] Block Definition
	- [ ] Line-Ellimination
	- [ ] Determine terminal state
	- [ ] Move-down

## Documentation

- State: n\*m + 11
	- Board State : n\*m
	- Block State : 10 (type(7),rotation,x,y)
		- When building inputs, the type will be converted into one-hot encoding
	- Alternative Block : 1 (type)

- Action: 6
	- Left, Right, Down, Rotate, Drop, Change

---

## Results
