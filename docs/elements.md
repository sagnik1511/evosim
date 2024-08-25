# Game Elements

The Game elements will interact with the agents and show progress of the current game.

The game elements only occupy a single cell and have different properties.

Currently we have 2 types of elements in the game.
- Resource
- Obstacle


### Resource : 

Resources are consumable, so any agent can increase their hp by consuming an resource.

- Wood : 
    - `hp` : 5
    - `movable` : No


### Obstacle:

Obstacles are non-consumable objects. They work as hurdles in the way.


- Rock :
    - `movable`: No