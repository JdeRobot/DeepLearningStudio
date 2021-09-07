---
title: "Inroducing Deep Learning Studio"
excerpt: "Collection of Deep Learning agorithms!"
usemathjax: true
sidebar:
  nav: "docs"

toc: true
toc_label: "Contents"
toc_icon: "cog"


categories:
- DeepLearningStudio
tags:
- Jderobot
- Deep Learning Studio
- DeepLearningStudio

author: Utkarsh Mishra
pinned: false
---

This has been a great journey and so far. Below you will find a concise summary of my contribution to the JdeRobot community as a Google Summer of Code student.
{: .text-justify}

## Objectives

- [x] PyTorch Extension
- [x] Randomization and Visualization
- [x] DeepLearningStudio: PilotNet base and stacked setups + DeepPilot CNN
- [x] Iris drone into Deep Learning Studio

## PyTorch Extension to Deep Learning Studio

Before GSoC 2021, Deep Learning Studio used to support Tensorflow based brains. My contributions included implementing and extending all the necessary components to support PyTorch based brains. This started from the community bonding period and I was able to implement both the Deep Learning and Reinforcement learning algorithms. You can refer to the following:
{: .text-justify}
- [Preliminary Code](https://github.com/TheRoboticsClub/gsoc2021-Utkarsh_Mishra/tree/community_bonding)
- [Bonding Blog Post 1](https://theroboticsclub.github.io/gsoc2021-Utkarsh_Mishra/gsoc/Community-Bonding-Week-1/)
- [Bonding Blog Post 2](https://theroboticsclub.github.io/gsoc2021-Utkarsh_Mishra/gsoc/Community-Bonding-Week-2/)

![]({{ site.url }}{{ site.baseurl }}/assets/images/blogs/fullimg_v1/pilotnet-h3.gif){: .center-image}
![]({{ site.url }}{{ site.baseurl }}/assets/images/blogs/drones_v1/takeoff-explicit.gif){: .center-image}

## Randomization and Visualization in Deep Learning Studio

Randomization for various initial positions in script mode was extended to GUI method and the same was done for the visualization of the Deep Learning Studio. This was done by implementing the randomization module separately and extending the existing code to better remote saving of visualized performance plots. Further, the information and custom hyperparameter flow from config files to respective brains was origanized and structured.   You can refer to the following:
{: .text-justify}
- [Deep Learning Studio PR #182, #183, #184, #185](https://github.com/JdeRobot/DeepLearningStudio)
- [Blog Post 2](https://theroboticsclub.github.io/gsoc2021-Utkarsh_Mishra/gsoc/Coding-Period-Week-2/)
- [Blog Post 4](https://theroboticsclub.github.io/gsoc2021-Utkarsh_Mishra/gsoc/Coding-Period-Week-4/)

![]({{ site.url }}{{ site.baseurl }}/assets/images/blogs/augs_v1/case1.png)
![]({{ site.url }}{{ site.baseurl }}/assets/images/blogs/analysis_v1/path_followed.png)

## DeepLearningStudio: PilotNet base and Stacked setups + DeepPilot CNN

This was the most significant segment for this GSoC period. I was able to implement the PilotNet algorithm in PyTorch. This, along with the existing tensorflow version of the algorithm, was used to start the JdeRobot - DeepLearningStudio. Further, an extension of the base code was used to explore temporal relations without memory based LSTM algorithms. The stacked brain was a success however further modifications are still going on and is a future work. Further, DeepPilot CNN was implemented in order to conduct extensive DL experiments on drone tasks. You can refer to the following: 
{: .text-justify}
- [DeepLearningStudio](https://github.com/JdeRobot/DeepLearningStudio/tree/main/Formula1-FollowLine)
- [Blog Post 1](https://theroboticsclub.github.io/gsoc2021-Utkarsh_Mishra/gsoc/Coding-Period-Week-1/)
- [Blog Post 3](https://theroboticsclub.github.io/gsoc2021-Utkarsh_Mishra/gsoc/Coding-Period-Week-3/)
- [Blog Post 6](https://theroboticsclub.github.io/gsoc2021-Utkarsh_Mishra/gsoc/Coding-Period-Week-6/)
- [Blog Post 7](https://theroboticsclub.github.io/gsoc2021-Utkarsh_Mishra/gsoc/Coding-Period-Week-7/)


<iframe width="560" height="315" src="https://www.youtube.com/embed/pbYfdXvtRLo" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


## Iris Drone into Deep Learning Studio

Finally, Behvaior Metrics is supposed to be a framework to test the performance of all kinds of machine learning based brains. Thus, the iris drone support was added so that DeepPilot CNN based models can be tested as well. I took the help of [JdeRobot/drones](https://github.com/JdeRobot/drones) to integrate the iris drone with the line folllowing task. First an explicit brain was developed which operates on complete visual control  based on only the frontal image. The explicit brain was modeled for 3 degrees of freedom, linear velocity, angular velocity and z-velocity. You can refer to the following: 
{: .text-justify}
- [Deep Learning Studio PR #193, #197, #201](https://github.com/JdeRobot/DeepLearningStudio)
- [DeepLearningStudio](https://github.com/JdeRobot/DeepLearningStudio/tree/main/Drone-FollowLine)
- [Blog Post 4](https://theroboticsclub.github.io/gsoc2021-Utkarsh_Mishra/gsoc/Coding-Period-Week-4/)
- [Blog Post 5](https://theroboticsclub.github.io/gsoc2021-Utkarsh_Mishra/gsoc/Coding-Period-Week-5/)
- [Blog Post 6](https://theroboticsclub.github.io/gsoc2021-Utkarsh_Mishra/gsoc/Coding-Period-Week-6/)

<iframe width="560" height="315" src="https://www.youtube.com/embed/GZs6OIQ_az0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>