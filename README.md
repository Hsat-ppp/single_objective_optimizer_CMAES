# single_objective_optimizer_CMAES
This is a single objective optimizer repository based on CMA-ES and dd-CMA-ES, its advanced version. Please refer to the original paper [1], [2] for detail.<br>
Please also refer to [4] for implementation by authors.
2022/06/28<br>
I added GA optimizer for comparison. Please refer to paper [3] for detail.<br>

## How to run
Please run main.py as:

```linux cui
python main.py
```

## References
[1] Nikolaus Hansen, "The CMA Evolution Strategy: A Tutorial", hal-01297037 HAL, 2005.<br>
[2] Y. Akimoto and N. Hansen, “Diagonal Acceleration for Covariance Matrix Adaptation Evolution Strategies,” Evol. Comput., vol. 28, no. 3, pp. 405–435, 2019.<br>
[3] 小林 重信, "実数値GAのフロンティア", 人工知能学会論文誌 24 (1), 147-162, 2009.<br>
[4] https://gist.github.com/youheiakimoto/1180b67b5a0b1265c204cba991fa8518