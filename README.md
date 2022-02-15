# PPO
Este repositorio es una implementación funcional lo más simple posible del algoritmo [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) con fines educativos. Esta implementación fue hecha tomando como base el [blog](https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8) de Eric Yang Yu @ UC San Diego.

A continuación se presenta una explicación del código presente en el archivo ppo.py comparado con cada linea del [pseudocódigo](https://www.researchgate.net/figure/PPO-Clip-Pseudocode-Implementation_fig22_335328616)

## Pseudocódigo:

<img src=PPO-Clip-Pseudocode.png align="center"/>

| Linea del codigo y breve descripcion | Metodo de la clase PPO o linea de codigo en que se implementa |
| ------------------------------------ | ------------------------------------------------------------- |
| 1. Se inicializan los parámetros (pesos o weights) del actor y critic | Método constructor, lineas self.actor y self.critic inicializan redes de PyTorch con pesos escogidos de manera aleatoria con una distribucion default de PyTorch |
| 2. Se designa un total de timesteps y se empieza a iterar el algoritmo hasta que se rebase el total | En el método learn, while timesteps_so_far < total_timesteps: |
| 3. Coleccion de batches s<sub>t</sub>, a<sub>t</sub>, &pi;<sub>&theta;</sub>(s<sub>t</sub>\|a<sub>t</sub>) | Método rollout |
| 4. Computo de los retornos estimados con discount factor &gamma; | Método compute_rtgs que es subrutina de rollout |
| 5. Calculo de la Advantage Function (ventaja de realizar una accion particular en un estado dado) A(a<sub>t</sub>, s<sub>t</sub>) = Q(a<sub>t</sub>, s<sub>t</sub>) - V(s<sub>t</sub>) | Método evualuate para estimar V y calculo de A dentro del método learn |
| 6. Actualización de los pesos de la red del actor optimizando una función sustituta (el corazón de PPO) | Se itera sobre self.n_updates_per_iteration (n° de épocas). En cada iteración la expresión que se optimiza es el promedio de un tensor que contiene los minimos entre la razón (&pi;<sub>&theta;</sub>/&pi;<sub>&theta;<sub>k</sub></sub>)A<sup>&pi;<sub>&theta;<sub>k</sub></sub></sup> y la misma razón cortada en un rango &epsilon; (por lo que esta versión de PPO se denomina PPO-clip) |
| 7. Se optimizan los pesos de la red que aproxima la Value function (el crítico) por el método de mínimos cuadrados (equivalentemente minimizando el cuadrado de la norma l2) | Dentro del iterador del paso 6, inmediatamente después actualizar los pesos del actor |
| 8. Nos aseguramos que el ciclo del paso 2 termine actualizando el numero de timesteps en cada paso | Ultima linea del método learn |
