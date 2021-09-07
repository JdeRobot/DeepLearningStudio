---
permalink: /documentation/
title: "Documentation"

sidebar:
  nav: "docs"

toc: true
toc_label: "Documentation"
toc_icon: "cog"

gallery1:
  - url: /assets/images/behavior_suite_diagram.png
    image_path: /assets/images/behavior_suite_diagram.png
    alt: ""
---


## Motivation

DeepLearningStudio is created as a significant segment for one of GSoC'21 projects ([more information](https://theroboticsclub.github.io/gsoc2021-Utkarsh_Mishra)). The project included implementing the PilotNet algorithm in PyTorch. This, along with the existing tensorflow version of the algorithm ([more information](https://roboticslaburjc.github.io/2019-phd-sergio-paniego/)), was used to start the JdeRobot - DeepLearningStudio tool. Further, an extension of the base code was used to explore temporal relations without memory based LSTM algorithms. The stacked brain was a success however further modifications are still going on and is a future work. Further, DeepPilot CNN was implemented in order to conduct extensive DL experiments on drone tasks. Parallely, the tensorflow version of all these algorithms are also being explored and derived from previous works by collaboarators ([more information](https://roboticslaburjc.github.io/2019-phd-sergio-paniego/)).

## Progress

The current status of the project contains the following:

### Formula-1 follow line algorithms

PilotNet [1] implementation for both the PyTorch and Tensorflow versions are implemented. This is accompanied by the memory less Stacked PilotNet version in PyTorch and memory based LSTM PilotNet version in Tensorflow. The code is trained on the [datasets](/quick_start/datasets) and validated on [BehaviorMetrics](https://github.com/JdeRobot/BehaviorMetrics). Some interesting results can be found in [channel](https://www.youtube.com/channel/UCgmUgpircYAv_QhLQziHJOQ).

### Iris drone follow line algorithms



## References

1. Bojarski, Mariusz, Davide Del Testa, Daniel Dworakowski, Bernhard Firner, Beat Flepp, Prasoon Goyal, Lawrence D. Jackel et al. "End to end learning for self-driving cars." arXiv preprint arXiv:1604.07316 (2016). [https://arxiv.org/abs/1604.07316](https://arxiv.org/abs/1604.07316)

```
@article{bojarski2016end,
  title={End to end learning for self-driving cars},
  author={Bojarski, Mariusz and Del Testa, Davide and Dworakowski, Daniel and Firner, Bernhard and Flepp, Beat and Goyal, Prasoon and Jackel, Lawrence D and Monfort, Mathew and Muller, Urs and Zhang, Jiakai and others},
  journal={arXiv preprint arXiv:1604.07316},
  year={2016}
}

@article{bojarski2017explaining,
  title={Explaining how a deep neural network trained with end-to-end learning steers a car},
  author={Bojarski, Mariusz and Yeres, Philip and Choromanska, Anna and Choromanski, Krzysztof and Firner, Bernhard and Jackel, Lawrence and Muller, Urs},
  journal={arXiv preprint arXiv:1704.07911},
  year={2017}
}
```

2. Rojas-Perez, L.O., & Martinez-Carranza, J. (2020). DeepPilot: A CNN for Autonomous Drone Racing. Sensors, 20(16), 4524. [https://doi.org/10.3390/s20164524](https://doi.org/10.3390/s20164524)

```
@article{rojas2020deeppilot,
  title={DeepPilot: A CNN for Autonomous Drone Racing},
  author={Rojas-Perez, Leticia Oyuki and Martinez-Carranza, Jose},
  journal={Sensors},
  volume={20},
  number={16},
  pages={4524},
  year={2020},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```


