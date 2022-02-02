from networks import FeedForwardNN

class PPO:
    def __init__(self, env):
        #Informacion del environment
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        #ALGO paso 1
        #Inicializar redes neuronales para el actor y critic (policy and value estimator)
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)


    def learn(self, total_timesteps):
        timesteps_so_far = 0
        while timesteps_so_far < total_timesteps:       #ALGO paso 2
            #ad