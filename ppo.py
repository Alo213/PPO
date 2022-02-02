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
        self._init_hyperparameters()

        #TODO change fill_value
        self.cov_var = torch.full(size=(self.act_dim,), fill_value = 0.5)

        self.cov_mat = torch.diag(self.cov_var)

    def _init_hyperparameters(self):
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600

    def rollout(self):
        #ALGO 3, collect data from set of trajectories...
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rews_to_go = []
        batch_lens = []

        obs = self.env.reset()
        done = False

        t = 0
        while t < self.timesteps_per_batch:
            ep_rews = []
            obs = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                t+= 1

                #Collect obs
                batch_obs.append(obs)
                
                action, log_prob = self.get_action(obs)    #Importante
                obs, rew, done, _ = self.env.step(action)

                #Collect action, log prob, rew
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break
            
            #Collect ep_length and episode rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)


    def get_action(self, obs):
        #Calcula accion tomada por el actor, se llama mean porque es la media de nuestra distribucion (para efectos de exploracion)
        mean = self.actor.forward(obs)

        dist = MultivariateNormal(mean, self.cov_mat)

        #Samplea una accion de la distribucion (solo para training, en testing se usa la accion mean, sin varianza) y toma su log probabilidad
        action = dist.sample()
        log_prob = dist.log_prob(action)

        #TODO revisar detach(), computation graphs
        # Note that I'm calling detach() since the action and log_prob  
        # are tensors with computation graphs, so I want to get rid
        # of the graph and just convert the action to numpy array.
        # log prob as tensor is fine. Our computation graph will
        # start later down the line.

        return action.detach().numpy(), log_prob.detach()



        

            

    def learn(self, total_timesteps):
        timesteps_so_far = 0
        while timesteps_so_far < total_timesteps:       #ALGO paso 2
            