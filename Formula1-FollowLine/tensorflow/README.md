# Tensorflow models

Models for both approaches that work for all the circuits in [BehaviorMetrics](https://github.com/JdeRobot/BehaviorMetrics) can be downloaded from here.

## Old dataset


| Model      | Download link |
| ----------- | ----------- |
| Deepest LSTM TinyPilotNet      | [link](https://drive.google.com/file/d/1Tzen7fSIs3hh9xir2J-NSu0XaVEmdewc/view?usp=sharing)       |
| PilotNet   | [link](https://drive.google.com/file/d/1CWVEKNqUPLvZ6L0nKzmonzLRa7i-lHUy/view?usp=sharing)        |


## OpenCV (Explicit 2) dataset

| Model                     | Memory | Download link |
|---------------------------| -----------| ----------- |
| PilotNet                  |NO | [link](https://drive.google.com/file/d/15b7W1kP0utLnc1olB1PD3-7Gll7nXgSy/view?usp=sharing)        |
| PilotNet (Optimized)       |NO | [link](https://drive.google.com/drive/folders/1j2nnmfvRdQF5Ypfv1p3QF2p2dpNbXzkt?usp=sharing)        |
| Deepest LSTM TinyPilotNet | NO | [link](https://drive.google.com/file/d/1M_nW37aPXUzbiG1Y2rw6DA0AOR64wbnD/view?usp=sharing)       |
| DeepestConvLSTMConv3DPilotnet     | YES x3 | [link](https://drive.google.com/file/d/1v8zN6TNOnJKUuyKq9S7fF7pf4pGTfraa/view?usp=sharing)       |
| PilotNetx3          | YES x3| [link](https://drive.google.com/file/d/1MsJEpxOQA7nEVejJBnSAoLr8R3oOkrJm/view?usp=sharing)       |

### With weather changes albumentations in training

| Model                        | Memory | Download link |
|------------------------------| -----------| ----------- |
| PilotNet (weather changes)   |NO | [link](https://drive.google.com/file/d/1OBH6589N2lgepNOneaKdO3Gb9mxplrPB/view?usp=sharing)        |
| Deepest LSTM TinyPilotNet (weather changes)   | NO |       |
| DeepestConvLSTMConv3DPilotnet (weather changes)| YES x3 | [link](https://drive.google.com/file/d/1DEvfjvErIJYdrfRT2asmciGs4Wwnubxa/view?usp=sharing)       |
| PilotNetx3 (weather changes)                  | YES x3| [link](https://drive.google.com/file/d/1_zGqi94OlOwDK3c_0WsXL8KMGyChcsTt/view?usp=sharing)       |

### With weather changes  and affine albumentations in training
| Model                                                      | Memory | Download link |
|------------------------------------------------------------| -----------| ----------- |
| PilotNet (weather changes and affine)                                 |NO |        |
| Deepest LSTM TinyPilotNet (weather changes and affine)                | NO |       |
| DeepestConvLSTMConv3DPilotnet (weather changes and affine) | YES x3 | [link](https://drive.google.com/file/d/1-eq1085wB3LInUMqpCZrFj878x6T7nW_/view?usp=sharing)       |
| PilotNetx3 (weather changes and affine)                               | YES x3|        |


### Performance of  optimized PilotNet networks

Method  | Model size (MB) | MSE  | Inference time (s) | Download link
--- | --- | --- | --- | ---
PilotNet (original tf format) | 195 | 0.041 | 0.0364 | [link](https://drive.google.com/file/d/15b7W1kP0utLnc1olB1PD3-7Gll7nXgSy/view?usp=sharing)
Baseline (tflite format)| 64.9173469543457 | 0.04108056542969754 | 0.007913553237915039 | [link](https://drive.google.com/file/d/1VYkX9938_fDKtzess4iWisryFxFKy6FW/view?usp=sharing)
Dynamic Range Q | **16.242530822753906** | **0.04098070281274293** | 0.004902467966079712 | [link](https://drive.google.com/file/d/1byV8yAzoxwk-Cj8FfmtSSBK2Lfbx6KQ3/view?usp=sharing)
Float16 Q | 32.464256286621094 | 0.041072421023905605 | 0.007940708875656129 | [link](https://drive.google.com/file/d/1Dq217ZVT60kpVbtEHnqx3hQZconO7CQp/view?usp=sharing)
Q aware training | 16.250564575195312 | 0.042138221871067326 | 0.009550530910491944 | [link](https://drive.google.com/file/d/1jSeQijCt7Uzg5d3JhfuDZyf3so2MkGO0/view?usp=sharing)
Weight pruning | 64.9173469543457 | 0.04257505210072217 | 0.0077278904914855956 | [link](https://drive.google.com/file/d/1DTlqzicJ1ZhBGGpXPD9vpfwnJZSJSwLv/view?usp=sharing)
Weight pruning + Q | **16.242530822753906** | 0.042606822364652304 | 0.004810283422470093 | [link](https://drive.google.com/file/d/1DTlqzicJ1ZhBGGpXPD9vpfwnJZSJSwLv/view?usp=sharing)
Integer only Q | 16.244918823242188 | 28157.721509850544 | 0.007908073902130127 | [link](https://drive.google.com/file/d/14IS6ZG_cKpYIfM6magC8z8XdyyLNMRa1/view?usp=sharing)
Integer (float fallback) Q | 16.244888305664062 | 0.04507085706016211 | 0.00781548523902893 | [link](https://drive.google.com/file/d/1VT5I30lBlObjIXnU424zZZJXQxQSPvif/view?usp=sharing)
CQAT | 16.250564575195312 | 0.0393811650675438 | 0.007680371761322021 | [link](https://drive.google.com/file/d/1N4WPaO4Uw5kPcOOZ6-AmVpU9IfYA08a7/view?usp=sharing)
PQAT | 16.250564575195312 | 0.043669467093106665 | 0.007949142932891846 | [link](https://drive.google.com/file/d/1RZkVP2ocT0dKGVLfF6Q1VZhsJWnJv1qY/view?usp=sharing)
PCQAT | 16.250564575195312 | 0.039242053481006144 | 0.007946955680847167 | [link](https://drive.google.com/file/d/1vMpOad4yGbubjd-Ry0JlU17jjop3McTy/view?usp=sharing)

*Q = Quantization*
*All the results are for models converted to tflite models if not specified.* <br>