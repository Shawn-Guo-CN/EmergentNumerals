---
typora-root-url: ./
---

# Gathering Game

### Diagram
![Diagram](/Diagram.JPG)

### Game Components
1. **Goal**: the goal is to keep foods in warehouse to *exact* numbers, e.g., in the above diagram, there are 3 kinds of foods, apple, banana and kiwi, the goal in *every* episode is to store 5 apples, 5 bananas and 5 kiwi; however, in different episode, the warehouse may be initialised to have different random numbers of foods, e.g. in the above diagram, agents are required to *cooperate* to pick-up 2 apples and 1 kiwi in the Food Gridworld.
2. **Food Warehouse**: there are always *3* kinds of foods in a *fixed* sequence, i.e. it is always described by a *triple (number of apples, number of bananas, number of kiwis)*, there at most 5  of one specific food, the triple would be initialised randomly in the beginning of an episode.
3. **Food Keeper Agent**: food keeper agent is the one that *can* check status of warehouse and send a message  to the other, it however *can not* get into the grid world and collect food.
4. **Food Gatherer Agent**: food gatherer agent is the one that first receives message from keeper and then can *pick-up* foods in the Food Gridworld.
5. **Food Gridworld**: there is only *one* gird with 4 actions: i) apple, pick-up an apple and put it into *knapsack*; ii) banana, similar to apple; iii) kiwi, similar to apple; iv) end, end the episode and return the knapsack to warehouse.
6. **Message**: message consist of  *at most 3* discrete symbols from a vocabulary containing *5* discrete symbols. (Hopefully, every symbol can correspond to a specific number in 1..5).

### Games as a MDP
1. **States (Environment)**: numbers of foods in the warehouse, message omitted by keeper, numbers of foods that are contained in knapsack;
2. **Observations**:
  1. Keeper: numbers of foods in the warehouse
  2. Gatherer: current status of knapsack (quite like a memory mechanism), message omitted by keeper
3. **Actions**: apple, banana, kiwi, end;
4. **Rewards**:  if gatherer choose to end the episode with *knapsack* containing exact numbers of food that can fulfill the warehouse, both agents receive a reward of 10, or -10 if knapsack contains fewer or more numbers of foods; during every step, reward to any action is -1. For instance, gatherer needs to take the end action with a knapsack containing *exactly* 2 apples and 1 kiwi in the above episode.

### Game Procedure
1. Food Keeper Agent observes the status of Food Warehouse;
2. Food Keeper Agent send message to Food Gatherer Agent;
3. Foog Gatherer Agents gathers foods in Food Gridworld, which may taks a few steps to complete.

### Definition of Numerals
With the assumption that all the information need to synchronised between two agents are is the numbers of every kind of food, we argue that all the symbols combinations appear as messages should all be combinations of numerals. Further, ideally, they could assign a symbol to a specific number. (Basically, what I  want to discuss here is that numerals are symbols being mapped to concepts of numbers. But I also worry that concepts of numbers here are actually from our human beings.)

From a functional perspective, numerals emerged in this game are to describe how many times an action should be executed. Take number $3$ for example, its function is to compress a string consists of 3 same symbols into 2 tokens, i.e.

$$3(apple) = apple \ \ apple\ \ apple$$ .

### Explanation about the Emergence of Numerals
Obviously, there are $5^3=125$ combinations of all kind of foods. However, with a vocabulary containing only 5 disrete symbols, it is far from possible to squeeze a specific status of warehouse into 1 token, agents neend at least 3 symbols to decribe the current warehouse. Without a limitition on the length of messages, they may omit repeating symbols in a single episode, which need to be checked by experiments.