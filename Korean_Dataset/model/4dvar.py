import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

##########################################################################
########################## LORENZ-63 MODEL PART ##########################
##########################################################################

class Lorenz63Model:
    def __init__(self, sigma=10.0, rho=28.0, beta=8./3, dt=0.01):
        """Initialise le modèle Lorenz-63 avec les paramètres par défaut ou spécifiés.
        
        Paramètres :
        - sigma, rho, beta : paramètres du système de Lorenz.
        - dt : pas de temps pour l'intégration numérique.
        """
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        self.nx = 3  # Nombre de variables d'état
        self.reset()

    def reset(self):
        """Réinitialise les variables d'état et les séries temporelles."""
        self.xvar = np.zeros(self.nx)
        self.xvar_series = []
        self.time_series = []
        self.time = 0.

    def rhs(self, x):
        """Calcul du terme de droite (Right-Hand Side) de l'équation de Lorenz.
        
        Paramètre :
        - x : l'état actuel du système.
        
        Retourne :
        - dxdt : la dérivée de l'état du système.
        """
        dxdt = np.zeros_like(x)
        dxdt[0] = self.sigma * (x[1] - x[0])
        dxdt[1] = x[0] * (self.rho - x[2]) - x[1]
        dxdt[2] = x[0] * x[1] - self.beta * x[2]
        return dxdt

    def rhs_tl(self, x, dx):
        """Calcul du modèle linéarisé tangent pour des perturbations.
        
        Paramètres :
        - x : l'état de référence pour la linéarisation.
        - dx : la perturbation de l'état.
        
        Retourne :
        - dxdt_tl : la dérivée de la perturbation.
        """
        dxdt_tl = np.zeros_like(dx)
        dxdt_tl[0] = self.sigma * (dx[1] - dx[0])
        dxdt_tl[1] = self.rho * dx[0] - dx[1] - x[0] * dx[2] - dx[0] * x[2]
        dxdt_tl[2] = x[1] * dx[0] + x[0] * dx[1] - self.beta * dx[2]
        return dxdt_tl

    def rhs_adj(self, x, adj):
        """Calcul du modèle adjoint pour des variables adjointes.
        
        Paramètres :
        - x : l'état de référence pour la linéarisation.
        - adj : la variable adjointe.
        
        Retourne :
        - adj_out : la dérivée de la variable adjointe.
        """
        adj_out = np.zeros_like(adj)
        adj_out[0] = -self.sigma * adj[0] + (self.rho - x[2]) * adj[1] + x[1] * adj[2]
        adj_out[1] = self.sigma * adj[0] - adj[1] + x[0] * adj[2]
        adj_out[2] = -x[0] * adj[1] - self.beta * adj[2]
        return adj_out

    def forward(self, initial_state, n_steps):
        """Exécute le modèle en avant sur un nombre spécifié de pas de temps à partir d'un état initial.
        
        Paramètres :
        - initial_state : l'état initial du système.
        - n_steps : le nombre de pas de temps à exécuter.
        """
        self.reset()
        self.xvar = np.copy(initial_state)
        for _ in range(n_steps):
            self.xvar_series.append(np.copy(self.xvar))
            self.time_series.append(self.time)
            self.xvar += self.rhs(self.xvar) * self.dt
            self.time += self.dt
        self.xvar_series = np.array(self.xvar_series).T

    def forward_tl(self, xref, dx, n_steps):
        """Propage une perturbation à travers le modèle linéarisé tangent.
        
        Paramètres :
        - xref : l'état de référence pour la linéarisation.
        - dx : la perturbation initiale.
        - n_steps : le nombre de pas de temps pour la propagation.
        
        Retourne :
        - dx_tl : la perturbation propagée.
        """
        dx_tl = np.copy(dx)
        for _ in range(n_steps):
            dx_tl += self.rhs_tl(xref, dx_tl) * self.dt
        return dx_tl

    def backward_adj(self, xref, adj, n_steps):
        """Propage une variable adjointe à travers le modèle adjoint en arrière dans le temps.
        
        Paramètres :
        - xref : l'état de référence pour la linéarisation.
        - adj : la variable adjointe initiale.
        - n_steps : le nombre de pas de temps pour la propagation.
        
        Retourne :
        - adx : la variable adjointe propagée.
        """
        adx = np.copy(adj)
        for _ in range(n_steps):
            adx += self.rhs_adj(xref, adx) * (-self.dt)
        return adx

    def plot(self):
        """Trace les séries temporelles des variables du modèle."""
        plt.figure(figsize=(12, 8))
        for ix in range(self.nx):
            plt.subplot(3, 1, ix + 1)
            plt.plot(self.time_series, self.xvar_series[ix, :], 'k', linewidth=1)
            plt.ylabel(f'Variable {ix+1}')
        plt.xlabel('Temps')
        plt.show()

##########################################################################
########################## DATA ASSIMILATION PART ##########################
##########################################################################

ntime = 200
lorenz_ref = Lorenz63Model()
xtrue = np.array([1.5, -1.5, 20.])
lorenz_ref.forward(xtrue, ntime)
lorenz_ref.plot()
true_state = lorenz_ref.xvar_series
true_time = lorenz_ref.time_series

# Définition des types et moments des observations
nvarobs = [0, 2]  # Variables observées
nobs = len(nvarobs)  # Nombre de variables observées
assim_steps = range(0, 10, 2)  # Étapes d'assimilation
nassim = len(assim_steps)  # Nombre d'étapes d'assimilation
assim_windows = 10  # Fenêtre d'assimilation

# Opérateur d'observation
hobs = np.zeros((nobs, 3))
for i, var in enumerate(nvarobs):
    hobs[i, var] = 1

# Observations et matrice de covariance
yobs = hobs @ true_state[:, assim_steps]  # true_state doit être défini ailleurs
noisestd = 0.1  # Écart-type du bruit
Robsm1 = np.eye(nobs) / (noisestd**2)

# Perturbation des observations pour simuler des erreurs d'observation
noise = np.random.randn(*yobs.shape)
yobs += noise * noisestd

# Arrière-plan
xbkgd = np.array([-3., 1., 15.])  # État de fond initial
xstd = 4  # Écart-type pour l'état de fond
Bm1 = np.eye(3) / (xstd**2)

def CostFunction(x_in):
    """Calcule la fonction de coût pour l'assimilation de données.
    
    Arguments :
    x_in -- État actuel estimé pour l'optimisation.
    
    Retourne :
    Le coût total combinant les coûts d'arrière-plan et d'observation.
    """
    # Coût d'arrière-plan
    xx = xbkgd - x_in
    Jb = 0.5 * np.dot(xx, Bm1 @ xx)
    
    # Coût d'observation
    Jo = 0
    m = Model()
    m.forward(x_in, assim_windows)
    for i, step in enumerate(assim_steps):
        innov = yobs[:, i] - hobs @ m.xvar_series[:, step]
        Jo += 0.5 * np.dot(innov.T, Robsm1 @ innov)
    
    return Jb + Jo

def CostGrad(x_in):
    """Calcule le gradient de la fonction de coût pour l'optimisation.
    
    Arguments :
    x_in -- État actuel estimé pour l'optimisation.
    
    Retourne :
    Le gradient de la fonction de coût par rapport à l'état estimé.
    """
    # Gradient du coût d'arrière-plan
    gJb = -Bm1 @ (x_in - xbkgd)
    
    # Gradient du coût d'observation
    m = Lorenz63Model()
    m.forward(x_in, assim_windows)
    xadj = np.zeros_like(x_in)
    for i in reversed(range(nassim)):
        step = assim_steps[i]
        innov = yobs[:, i] - hobs @ m.xvar_series[:, step]
        xadj += hobs.T @ (Robsm1 @ innov)
        if i > 0:
            nsteps = step - assim_steps[i - 1]
        else:
            nsteps = step + 1
        xref = m.xvar_series[:, step - nsteps]
        xadj = m.backward_adj(xref, xadj, nsteps)
    
    gJo = -xadj
    return gJb + gJo

# Optimisation de l'état initial
res = minimize(CostFunction, xbkgd, method='L-BFGS-B', jac=CostGrad, options={'maxiter': 30})

# Visualisation des résultats
bkgd = Lorenz63Model()
bkgd.forward(xbkgd, assim_windows)
ana = Lorenz63Model()
ana.forward(res.x, assim_windows)

plt.figure(figsize=(12, 8))
for ix in range(3):
    plt.subplot(3, 1, ix + 1)
    plt.plot(bkgd.time_series[:assim_windows], bkgd.xvar_series[ix, :assim_windows], 'blue', linewidth=1., label='Background')
    plt.plot(true_time[:assim_windows], true_state[ix, :assim_windows], 'red', linewidth=1., label='Reference')
    if ix in nvarobs:
        obs_indices = [nvarobs.index(ix) for ix in nvarobs if ix == ix]
        plt.plot(assim_steps, yobs[obs_indices, :], 'ro', label='Obs')
    plt.plot(ana.time_series[:assim_windows], ana.xvar_series[ix, :assim_windows], 'black', linewidth=1., label='Analysis')
    plt.legend()
plt.show()