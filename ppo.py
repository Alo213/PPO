from networks import FeedForwardNN

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.optim import Adam

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
        self.actor_optim = Adam(self.actor.parameters(), lr = self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        #TODO change fill_value
        self.cov_var = torch.full(size=(self.act_dim,), fill_value = 0.5)

        self.cov_mat = torch.diag(self.cov_var)

    def _init_hyperparameters(self):
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.lr = 0.005
        self.gamma = 0.95
        self.n_updates_per_iteration = 5                          #Number of epochs, arbitrarily chosen
        self.clip = 0.2                                           

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []

        for ep_rews in reversed(batch_rews):

            discounted_reward = 0

            for rew in reversed(ep_rews):
                discounted_reward = rew + gamma*discounted_reward
                batch_rtgs.insert(0, discounted_reward)

        batch_rtgs = torch.tensor(batch_rtgs, dtype=float)

        return batch_rtgs

    def rollout(self):
        #ALGO 3, collect data from set of trajectories...
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
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
        
        #Turn batches to tensors before return to draw computation graphs later
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        batch_rtgs = self.compute_rtgs(batch_rews)                              #ALGO step 4

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic.forward(batch_obs).squeeze()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        return V, log_probs

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
        #asdfjhas


        

            

    def learn(self, total_timesteps):
        timesteps_so_far = 0
        while timesteps_so_far < total_timesteps:                                                   #ALGO paso 2

            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()         #ALGO step 3

            V, _ = self.evaluate(batch_obs)                                                         #ALGO step 5

            #Calculate A = Q - V_{phi_k}
            A_k = self.batch_rtgs - V.detach()                                                      #ALGO step 5

            #Normalize Advantages
            A_k = (A_k - A_k.mean())/(A_k.std() + 1e-10)                                            #Training trick

            for _ in range(self.n_updates_per_iteration):                                           #ALGO step 6
                #Calculate pi_theta(a_t|s_t) and ratio with pi_theta old
                V, current_log_probs = self.evaluate(batch_obs, batch_acts)
                ratios = torch.exp(current_log_probs - batch_log_probs)

                #Calculate surrogate losses 
                surr1 = ratios*A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1+ self.clip)*A_k
                actor_loss = (-torch.min(surr1,surr2)).mean()

                #Calculate gradients and perform backpropagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()                                                             #ALGO step 6

                critic_loss = nn.MSELoss()(V, batch_rtgs)                                           #ALGO step 7
                #Calculate gradients and perform backpropagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()                                                            #ALGO step 7


            #Calculate how many timesteps where collected this batch
            timesteps_so_far += np.sum(batch_lens)                                                  #ALGO step 8, end for


        




    
            