# PPO
Este repositorio es una implementación funcional lo más simple posible del algoritmo [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) con fines educativos. Esta implementación fue hecha tomando como base el [blog](https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8) de Eric Yang Yu @ UC San Diego.

A continuación se presenta una explicación del código presente en el archivo ppo.py comparado con cada linea del [pseudocódigo](https://www.researchgate.net/figure/PPO-Clip-Pseudocode-Implementation_fig22_335328616)

## Pseudocódigo:

<img src=PPO-Clip-Pseudocode.png align="center"/>

| Linea del codigo y breve descripcion | Metodo de la clase PPO o linea de codigo en que se implementa |
| ------------------------------------ | ------------------------------------------------------------- |
| 1. Se inicializan los parámetros del actor y critic | Método constructor, self.actor y self.critic inicializan redes de PyTorch con pesos escogidos por default |
