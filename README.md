# PPO
Este repositorio es una implementación funcional lo más simple posible del algoritmo [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) con fines educativos. Esta implementación fue hecha tomando como base el [blog](https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8) de Eric Yang Yu @ UC San Diego.

A continuación se presenta una explicación del código presente en el archivo ppo.py comparado con cada linea del [pseudocódigo](https://www.researchgate.net/figure/PPO-Clip-Pseudocode-Implementation_fig22_335328616)

## Pseudocódigo:

<img src=PPO-Clip-Pseudocode.png align="center"/>

| Linea del codigo y breve descripcion | Metodo de la clase PPO o linea de codigo en que se implementa |
| ------------------------------------ | ------------------------------------------------------------- |
| 1. Se inicializan los parámetros del actor y critic | Método constructor, lineas self.actor y self.critic inicializan redes de PyTorch con pesos escogidos de manera aleatoria con una distribucion default de PyTorch |
| 2. Se designa un total de timesteps y se empieza a iterar el algoritmo hasta que se rebase el total | En el método learn, while timesteps_so_far < total_timesteps: |
| 3. Coleccion de batches s<sub>t</sub>, a<sub>t</sub>, &pi;(s<sub>t</sub>\|a<sub>t</sub>) | Método rollout |
