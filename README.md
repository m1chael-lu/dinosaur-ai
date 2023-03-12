# dinosaur-ai #
Internet-down dinosaur game, but with neural networks and genetic algorithms. I found this project while browsing through middle school projects. 

<div align="center"> 
  <img src="Dinosaur game.gif"/>
</div>


## Training ##

- Designed a three layer feed forward neural network for each dinosaur species with a sigmoid function as an activation function. 
- Within each iteration, the "fitness" or optimizable score is based on the number of obstacle passed. 
- To start training, 100 random dinosaur species are created with randomized parameters
- Top species are kept and mutated to generate a new generation of dinosaurs. 
- The displayed video clip is one of the final iterations of the training

## Dependencies ##

- Python 3
- numpy
- pygame
