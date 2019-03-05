# Hyperparameter Configurations

---
CONF
 - num_food_types = 1
 - max_capacity = 5

Train Script
 - device = torch.device("cuda")
 - lr = 1e-3
 - test_interval = 20
 - decay_interval = 50
 - batch_size = 64
 - replay_pool = ReplayMemory(1000)
 - torch.manual_seed(1234)

Replay Memory
 - random.seed(1234)

ENV
 - torch.manual_seed(1234)

**Performance Record**
[test score]episode 20: -77.10
[test score]episode 40: -73.22
[test score]episode 60: -236.19
[test score]episode 80: 97.21
[test score]episode 100: 97.51
[test score]episode 120: 97.57
[test score]episode 140: 97.43
[test score]episode 160: 97.47
[test score]episode 180: 61.54
[test score]episode 200: 20.03
[test score]episode 220: 46.14
[test score]episode 240: 91.67
[test score]episode 260: 95.57
[test score]episode 280: 97.56
[test score]episode 300: 97.52
[test score]episode 320: 94.34
[test score]episode 340: 97.24
[test score]episode 360: 95.62
[test score]episode 380: 97.52
[test score]episode 400: 97.71

---
CONF
 - num_food_types = 1
 - max_capacity = 5

Train Script
 - device = torch.device("cuda")
 - lr = 1e-3
 - test_interval = 20
 - decay_interval = 50
 - batch_size = 64
 - replay_pool = ReplayMemory(1000)
 - torch.manual_seed(1234)

Replay Memory
 - random.seed(1234)

ENV
 - torch.manual_seed(1234)

**Performance Record**
[test score]episode 20: -77.98
[test score]episode 40: 9.75
[test score]episode 60: 82.52
[test score]episode 80: 95.39
[test score]episode 100: 97.54
[test score]episode 120: 97.38
[test score]episode 140: 95.46
[test score]episode 160: 97.52
[test score]episode 180: 97.45