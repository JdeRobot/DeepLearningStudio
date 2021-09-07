---
permalink: /install/

title: "Installation and use"

sidebar:
  nav: "docs"


gallery:
  - url: /install/ros_melodic
    image_path: /assets/images/install/ros_melodic.png
    alt: "ROS melodic"
    title: "ROS melodic"
  - url: /install/ros_noetic
    image_path: /assets/images/install/ros_noetic.jpg
    alt: "ROS noetic"
    title: "ROS noetic"

---

Best to setup a virtual environment with python 3.6

```bash
cd ~ && mkdir pyenvs && cd pyenvs
python3 -m pip install virtualenv
virtualenv dlstudio --python=python3

cd ~
git clone https://github.com/JdeRobot/DeepLearningStudio DeepLearningStudio
cd DeepLearningStudio
source ~/pyenvs/dlstudio/bin/activate
python3 -m pip install -r requirements.txt
```
