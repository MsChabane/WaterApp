import math
import numpy as np

class PO:
  def __init__(self,pop_size,max_iter,lb,ub,dim):
    self.pop_size=pop_size
    self.max_iter=max_iter
    self.lb=lb
    self.ub=ub
    self.dim=dim
    
  def solve(self, fobj):
      if np.isscalar(self.ub):
          self.ub = np.ones(self.dim) * self.ub
          self.lb = np.ones(self.dim) * self.lb
      # Initialization
      X0 =self.initialization()
      X = X0
      # Compute initial fitness values
      fitness = np.zeros(self.pop_size)
      for i in range(self.pop_size):
          fitness[i] = fobj(X[i, :])
      index = np.argsort(fitness)
      fitness = np.sort(fitness)
      GBestF = fitness[0]
      GBestX = X[index[0], :]
      X_new = X[index, :]
      X = X_new
      curve = np.zeros(self.max_iter)
     
      
      
      # Start search
      for i in range(self.max_iter):
          if i % 100 == 0 and i > 0:
              print(f'At iteration {i}, the fitness is {curve[i - 1]}')
          
          alpha = np.random.rand() / 5
          sita = np.random.rand() * np.pi
          for j in range(self.pop_size):
              St = np.random.randint(1, 6)
              if St == 1:
                  X_new[j, :] = (X[j, :] - GBestX) * self.levy() + np.random.rand() * np.mean(X, axis=0) * (
                              1 - i / self.max_iter) ** (2 * i / self.max_iter)
              elif St == 2:
                  X_new[j, :] = X[j, :] + GBestX * self.levy() + np.random.randn() * (1 - i / self.max_iter) * np.ones(self.dim)
              elif St == 3:
                  H = np.random.rand()
                  if H < 0.5:
                      X_new[j, :] = X[j, :] + alpha * (1 - i / self.max_iter) * (X[j, :] - np.mean(X, axis=0))
                  else:
                      X_new[j, :] = X[j, :] + alpha * (1 - i / self.max_iter) * np.exp(-j / (np.random.rand() * self.max_iter))
              else:
                  X_new[j, :] = X[j, :] + np.random.rand() * np.cos((np.pi * i) / (2 * self.max_iter)) * (
                              GBestX - X[j, :]) - np.cos(sita) * (i / self.max_iter) ** (2 / self.max_iter) * (X[j, :] - GBestX)

              # Boundary control
              X_new[j, :] = np.clip(X_new[j, :], self.lb, self.ub)

              # Finding the best location so far
              if fobj(X_new[j, :]) < GBestF:
                  GBestF = fobj(X_new[j, :])
                  GBestX = X_new[j, :]

          # Update positions and fitness
          fitness_new = np.array([fobj(ind) for ind in X_new])
          for s in range(self.pop_size):
              if fitness_new[s] < GBestF:
                  GBestF = fitness_new[s]
                  GBestX = X_new[s, :]
          X = X_new
          fitness = fitness_new

          # Sorting and updating
          index = np.argsort(fitness)
          fitness = np.sort(fitness)
          X = X[index, :]
          curve[i] = GBestF

      Best_pos = GBestX
      Best_score = GBestF
      return  Best_pos, Best_score, curve
  def levy(self):
      beta = 1.5
      sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (
              math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
      u = np.random.randn(1, self.dim) * sigma
      v = np.random.randn(1, self.dim)
      step = u / np.power(np.abs(v), (1 / beta))
      return step
  def initialization(self):
      return np.random.rand(self.pop_size, self.dim) * (self.ub -self. lb) + self.lb


#

class HHO:


    def __init__(self,population_size,max_iter,lb,ub,dim):
        self.dim = dim
        self.lb = np.array([lb] * dim) if isinstance(lb, (int, float)) else np.array(lb)
        self.ub = np.array([ub] * dim) if isinstance(ub, (int, float)) else np.array(ub)
        self.population_size = population_size
        self.max_iter = max_iter

    def update_position(self, hawk, rabbit, t, q, e):
        r1 = np.random.rand()
        r2 = np.random.rand()
        J = 2 * (1 - t / self.max_iter) * np.random.rand()
        E = 2 * e * r1 - e

        if abs(E) >= 1:
            # Exploration phase
            X_rand = self.population[np.random.randint(0, self.population_size)]
            new_hawk = X_rand - r2 * abs(X_rand - 2 * r1 * hawk)
        else:
            # Exploitation phase
            if q >= 0.5:
                # Soft besiege
                new_hawk = rabbit - E * abs(J * rabbit - hawk)
            else:
                # Hard besiege
                if np.random.rand() >= 0.5:
                    # Soft besiege with progressive rapid dives
                    new_hawk = rabbit - E * abs(rabbit - hawk) + np.random.normal(0, 1, self.dim)
                else:
                    # Hard besiege with progressive rapid dives
                    new_hawk = np.mean(self.population, axis=0) - E * abs(J * rabbit - hawk)

        # Bound position to search space
        return np.clip(new_hawk, self.lb, self.ub)

    def solve(self,obj_function):
        self.obj_function = obj_function
        
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        self.fitness = np.apply_along_axis(self.obj_function, 1, self.population)
        self.best_idx = np.argmin(self.fitness)
        self.best_position = self.population[self.best_idx]
        self.best_score = self.fitness[self.best_idx]
        self.history = np.zeros((self.max_iter))
        for t in range(self.max_iter):
            for i in range(self.population_size):
                hawk = self.population[i]
                q = np.random.rand()
                e = np.random.uniform(-1, 1)
                new_position = self.update_position(hawk, self.best_position, t, q, e)

                # Evaluate new position
                new_fitness = self.obj_function(new_position)

                # Update if the new position is better
                if new_fitness < self.fitness[i]:
                    self.population[i] = new_position
                    self.fitness[i] = new_fitness

                    # Update global best
                    if new_fitness < self.best_score:
                        self.best_score = new_fitness
                        self.best_position = new_position

            self.history[t]=self.best_score


        return self.best_position, self.best_score, self.history    
    
 #####
 


class RUN:
    def __init__(self,population_size,max_iter,lb,ub,dim):
        self.dim = dim
        self.lb = np.array([lb] * dim) if isinstance(lb, (int, float)) else np.array(lb)
        self.ub = np.array([ub] * dim) if isinstance(ub, (int, float)) else np.array(ub)
        self.population_size = population_size
        self.max_iter = max_iter

    def solve(self,obj_function):
        self.obj_function = obj_function
        
        self.a = 0.5 # Coefficient for the Runge-Kutta mechanism

        # Initialize positions randomly within bounds
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size,self. dim))
        self.fitness = np.apply_along_axis(self.obj_function, 1, self.population)

        # Track the best solution
        self.best_idx = np.argmin(self.fitness)
        self.best_position = self.population[self.best_idx]
        self.best_score = self.fitness[self.best_idx]

        # History of best fitness
        self.history = np.zeros((self.max_iter))
        for t in range(self.max_iter):
            for i in range(self.population_size):
                current_position = self.population[i]

                # Calculate four intermediate steps (Runge-Kutta inspired updates)
                k1 = self.a * np.random.uniform(-1, 1, self.dim)
                k2 = self.a * np.random.uniform(-1, 1, self.dim) + 0.5 * k1
                k3 = self.a * np.random.uniform(-1, 1, self.dim) + 0.5 * k2
                k4 = self.a * np.random.uniform(-1, 1, self.dim) + k3

                # Update position based on Runge-Kutta approximation
                new_position = current_position + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
                new_position = np.clip(new_position, self.lb, self.ub)  # Ensure bounds

                # Evaluate the new position
                new_fitness = self.obj_function(new_position)

                # Update the population and personal bests
                if new_fitness < self.fitness[i]:
                    self.population[i] = new_position
                    self.fitness[i] = new_fitness

                    # Update global best if necessary
                    if new_fitness < self.best_score:
                        self.best_score = new_fitness
                        self.best_position = new_position

            # Record the best score of the current iteration
            self.history[t]=self.best_score


        return self.best_position, self.best_score, self.history


class SMO:
    def __init__(self,population_size,max_iter,lb,ub,dim):
        self.dim = dim
        self.lb = np.array([lb] * dim) if isinstance(lb, (int, float)) else np.array(lb)
        self.ub = np.array([ub] * dim) if isinstance(ub, (int, float)) else np.array(ub)
        self.population_size = population_size
        self.max_iter = max_iter

    def slime_mould_weight(self, rank):
        """Compute weight based on fitness rank."""
        return 1 / (rank + 1e-8)

    def solve(self,obj_function):
        self.obj_function = obj_function
        

        # Initialize population
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        self.fitness = np.apply_along_axis(self.obj_function, 1, self.population)

        # Best solution tracking
        self.best_idx = np.argmin(self.fitness)
        self.best_position = self.population[self.best_idx]
        self.best_score = self.fitness[self.best_idx]

        # History of best scores
        self.history = np.zeros((self.max_iter))
        for t in range(self.max_iter):
            # Rank fitness (lower is better)
            sorted_indices = np.argsort(self.fitness)
            ranked_fitness = self.fitness[sorted_indices]
            ranked_population = self.population[sorted_indices]

            # Update weights for the population
            weights = np.array([self.slime_mould_weight(i) for i in range(self.population_size)])

            for i in range(self.population_size):
                current = self.population[i]
                r = np.random.rand()
                weight = weights[i]

                if r < 0.5:
                    # Exploration phase
                    rand_idx = np.random.randint(0, self.population_size)
                    rand_pos = self.population[rand_idx]
                    new_position = current + weight * (rand_pos - current) * np.random.uniform(-1, 1, self.dim)
                else:
                    # Exploitation phase
                    best_weighted = self.best_position + weight * (self.best_position - current)
                    new_position = best_weighted + np.random.normal(0, 1, self.dim)

                # Clip to bounds and evaluate
                new_position = np.clip(new_position, self.lb, self.ub)
                new_fitness = self.obj_function(new_position)

                # Update if the new position is better
                if new_fitness < self.fitness[i]:
                    self.population[i] = new_position
                    self.fitness[i] = new_fitness

                    # Update global best if necessary
                    if new_fitness < self.best_score:
                        self.best_score = new_fitness
                        self.best_position = new_position

            # Update fitness and track history
            self.history[t]=self.best_score


        return self.best_position, self.best_score, self.history





class MRFO:
    def __init__(self,population_size,max_iter,lb,ub,dim):
        self.dim = dim
        self.lb = np.array([lb] * dim) if isinstance(lb, (int, float)) else np.array(lb)
        self.ub = np.array([ub] * dim) if isinstance(ub, (int, float)) else np.array(ub)
        self.population_size = population_size
        self.max_iter = max_iter

    def chain_foraging(self, position, leader, t):
        return position + np.random.uniform(-1, 1) * (leader - position) * np.tanh(t / self.max_iter)

    def somersault_foraging(self, position, global_best):
        return position + np.random.uniform(-1, 1) * (global_best - position)

    def solve(self,obj_function):
        self.obj_function = obj_function
        

        # Initialize population
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        self.fitness = np.apply_along_axis(self.obj_function, 1, self.population)

        # Track best solution
        self.best_idx = np.argmin(self.fitness)
        self.best_position = self.population[self.best_idx]
        self.best_score = self.fitness[self.best_idx]

        # History of best fitness values
        self.history = np.zeros((self.max_iter))
        for t in range(self.max_iter):
            leader = self.population[np.argmin(self.fitness)]

            for i in range(self.population_size):
                if np.random.rand() < 0.5:
                    # Chain foraging
                    new_position = self.chain_foraging(self.population[i], leader, t)
                else:
                    # Somersault foraging
                    new_position = self.somersault_foraging(self.population[i], self.best_position)

                # Clip the new position to bounds
                new_position = np.clip(new_position, self.lb, self.ub)

                # Evaluate the new fitness
                new_fitness = self.obj_function(new_position)

                # Update individual position and fitness if better
                if new_fitness < self.fitness[i]:
                    self.population[i] = new_position
                    self.fitness[i] = new_fitness

                    # Update global best if necessary
                    if new_fitness < self.best_score:
                        self.best_score = new_fitness
                        self.best_position = new_position

            # Update best leader for the next iteration
            self.history[t] =self.best_score

        return self.best_position, self.best_score, self.history



