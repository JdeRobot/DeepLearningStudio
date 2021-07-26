# DeepLearningStudio

## Information regarding this branch

This branch contains the deep learning regression and classification models.


## Structure of the branch

    ├── Formula1-FollowLine
        |
        |── PilotNet                                # Pilot Net pytorch implementation
        |   ├── scripts                             # scripts for running experiments 
        |   ├── utils                               
        |   |   ├── pilot_net_dataset.py            # Torchvision custom dataset
        |   |   ├── pilotnet.py                     # CNN for PilotNet
        |   |   ├── transform_helpers.py            # Data Augmentation
        |   |   └── processing.py                   # Data collecting, processing and utilities
        |   └── train.py                            # training code
        |
        └── PilotNetStacked                         # Pilot Net Stacked Image implementation
            ├── scripts                             # scripts for running experiments 
            ├── utils                               
            |   ├── pilot_net_dataset.py            # Sequentially stacked image dataset
            |   ├── pilotnet.py                     # Modified Hyperparams 
            |   ├── transform_helpers.py            # Data Augmentation
            |   └── processing.py                   # Data collecting, processing and utilities
            └── train.py                            # training code


## Setting up this branch

Best to setup a virtual environment with python 3.6

```
cd ~ && mkdir pyenvs && cd pyenvs
python3 -m pip install virtualenv
virtualenv dlstudio --python=python3

cd ~
git clone https://github.com/JdeRobot/DeepLearningStudio DeepLearningStudio
git checkout pilotnet
source ~/pyenvs/dlstudio/bin/activate
python3 -m pip install -r requirements.txt
```

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