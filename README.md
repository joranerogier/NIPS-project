  ## Experiment 1 
Given the features we selected, they can be used as state variables that are time independent.  We create an eleven dimension vector of numbers between zero and one that encodes these variables. 
And then every game can be seen as a sequence or path inside this vector field. 
The regard appears at the end of the game and for that reason, it is very sparse. 
The alternative we propose is to define the value of an action in the space (action, state) as a function of winning or losing the game. 
If the agent wins the actions it took over the game get higher (to the middle point between the previous value and 1) and if he looses then the actions he took over the game get lower (to the middle point between the previous value and 0).
The mapping of this action-value function is a multilayer dense network that gets trained after every game. 
