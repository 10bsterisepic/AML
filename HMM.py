import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
from sklearn.metrics import accuracy_score

states = ['Rainy', 'Sunny']
n_states = len(states)
en_states = {states:i for i, states in enumerate(states)}

obs = ['walk', 'shop', 'clean']
n_obs = len(obs)
en_obs = {obs:i for i, obs in enumerate(obs)}

states_seq = ["Sunny", "Rainy", "Rainy", "Rainy", "Sunny", "Sunny", "Rainy"]
en_states_seq = np.array([en_states[s] for s in states_seq])
actual_states_en = np.array([states_seq[s] for s in en_states_seq])

obs_seq = ['walk', 'shop', 'clean', 'clean', 'walk', 'walk', 'shop']
en_obs_seq = np.array([en_obs[o] for o in obs_seq]).reshape(-1, 1)

start_prob = np.array([0.6, 0.4])

trans_prob = np.array([[0.7, 0.3],
                      [0.4, 0.6]])

emission_prob = np.array([[0.1, 0.4, 0.5],
                         [0.6, 0.3, 0.1]])

model = hmm.CategoricalHMM(n_components = n_states, init_params="")
model.startprob_ = start_prob
model.transmat_ = trans_prob
model.emissionprob_ = emission_prob

logprob, pred_states = model.decode(en_obs_seq, algorithm = 'viterbi')
pred_states_names = [states[state] for state in pred_states]
accuracy = accuracy_score(en_states_seq, pred_states)

print(f'Most likely states by Viterbi: {pred_states_names}')
print(f'Actual States: {states_seq}')
print(f'Accuracy: {accuracy:.2f}')
print(f'Log Probability of sequence: {logprob:.2f}')

def plot_matrix(matrix, labels_row, labels_col, title):
    plt.figure(figsize=(6,4))
    sns.heatmap(matrix, annot = True, fmt='.2f', cmap = 'Blues',
                xticklabels = labels_col, yticklabels = labels_row)
    plt.title(title)
    plt.xlabel('To')
    plt.ylabel('From')
    plt.show()

plot_matrix(model.transmat_, states, states, 'State Transition Probabilities')

plot_matrix(model.emissionprob_, states, obs, 'Emission Probabilities')
