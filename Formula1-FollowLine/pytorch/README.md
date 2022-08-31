# PyTorch models

Models for both approaches that work for all the circuits in [BehaviorMetrics](https://github.com/JdeRobot/BehaviorMetrics) can be downloaded from here.


### Performance of  optimized PilotNet networks

Method | Model size (MB) | MSE | Inference time (s) | Download link
--- | --- | --- | --- | --- 
Baseline | 6.118725776672363 | 0.07883577048224175 | 0.002177743434906006 | [link](https://drive.google.com/file/d/1sZPDgfNzXbXvsmwiHfCHm0MR5FvESoXz/view?usp=sharing)
Dynamic Range Q | 1.9464960098266602 | 0.07840978354769981 | 0.003166124105453491 | [link](https://drive.google.com/file/d/18hdkseAezyiRWTofGYw2p8m7pdtKPa2P/view?usp=sharing)
Static Q | **1.6051549911499023** | 0.07881803711366263 | 0.0026564240455627442 | [link](https://drive.google.com/file/d/18hdkseAezyiRWTofGYw2p8m7pdtKPa2P/view?usp=sharing)
QAT | 1.606736183166504 | 0.07080468241058822 | 0.0027930240631103514 | [link](https://drive.google.com/file/d/15DIMuBIi49Tx5F4ckU0U-rWwfmN3ac38/view?usp=sharing)
Local Prune | 6.119879722595215 | 0.07294941230377715 | **0.0020925970077514647** | [link](https://drive.google.com/file/d/15DIMuBIi49Tx5F4ckU0U-rWwfmN3ac38/view?usp=sharing)
Global Prune | 6.119948387145996 | 0.07079961896774226 | 0.00215102481842041 | [link](https://drive.google.com/file/d/1cz7xBI6uvhWhTcbZUhz8hsRnmyKvWFgd/view?usp=sharing)
Prune + Quantization | 1.606797218322754 | **0.06724451272748411** | 0.002662529468536377 | [link](https://drive.google.com/file/d/1zRgAHGef3MeLxVw-8xzipW-ZGhJoHyJa/view?usp=sharing)

The original Pytorch model can be downloaded from here - [link](https://drive.google.com/file/d/1GFa8xe4XYh4xJQnFCNXcXDoeN1ykAKi1/view?usp=sharing)