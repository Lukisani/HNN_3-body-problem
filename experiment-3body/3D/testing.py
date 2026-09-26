##### ENERGY #####
def potential_energy(state):
    '''U=\sum_i,j>i G m_i m_j / r_ij'''
    tot_energy = np.zeros((1,1,state.shape[2]))
    for i in range(state.shape[0]):
        for j in range(i+1, state.shape[0]):
            r_ij = ((state[i:i+1,1:4] - state[j:j+1,1:4])**2).sum(1, keepdims=True)**0.5
            m_i = state[i:i+1,0:1]
            m_j = state[j:j+1,0:1]
            tot_energy += m_i * m_j / r_ij
    U = -tot_energy.sum(0).squeeze()
    return U

def kinetic_energy(state):
    '''T=\sum_i 0.5 * p^2 / m'''
    mass = state[:, 0:1]  # (3, 1)
    p = state[:, 4:7]     # (3, 3) [px, py, pz]
    energies = 0.5 * (p**2).sum(axis=1, keepdims=True) / mass
    T = energies.sum(0).squeeze()
    return T

def total_energy(state):
    return potential_energy(state) + kinetic_energy(state)