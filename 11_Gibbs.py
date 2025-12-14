import numpy as np
import matplotlib.pyplot as plt
docs = [['apple', 'banana', 'apple'],
        ['dog', 'cat', 'dog'],
        ['apple', 'dog', 'banana']]

words = []
for doc in docs:
    for word in doc:
        words.append(word)

vocab = list(set(words))
V=len(vocab)
word2id = {w: i for i, w in enumerate(vocab)}
id2word = {i:w for w, i in word2id.items()}
docs_id = [[word2id[w] for w in doc] for doc in docs]
D=len(docs_id)
K=2
alpha, beta = 0.5, 0.5

np.random.seed(42)
z=[]
ndk=np.zeros((D, K))
nkw=np.zeros((K, V))
nk=np.zeros(K)

for d, doc in enumerate(docs_id):
    topics=[]
    for w in doc:
        t=np.random.randint(K)
        topics.append(t)
        ndk[d, t]+=1
        nkw[t, w]+=1
        nk[t]+=1
    z.append(topics)
iters=200
trace_topic0=[]
trace_loglike=[]

def log_likelihood(ndk, nkw, nk, alpha, beta):
    """Rough log likelihood indicator for convergence."""
    ll=0
    for k in range(K):
        ll+=np.sum(np.log(nkw[k]+beta))-np.log(np.sum(nkw[k]+beta))
    for d in range(D):
        ll+=np.sum(np.log(ndk[d]+alpha))-np.log(np.sum(ndk[d]+alpha))
    return ll

for it in range(iters):
    for d, doc in enumerate(docs_id):
        for i, w in enumerate(doc):
            t=z[d][i]
            ndk[d, t]-=1
            nkw[t, w]-=1
            nk[t]-=1
            p=(ndk[d]+alpha)*(nkw[:, w]+beta)/(nk+V*beta)
            p/=np.sum(p)
            new_t=np.random.choice(K, p=p)

            z[d][i]=new_t
            ndk[d, new_t]+=1
            nkw[new_t, w]+=1
            nk[new_t]+=1
    trace_topic0.append(nk[0]/np.sum(nk))
    trace_loglike.append(log_likelihood(ndk, nkw, nk, alpha, beta))

theta = (ndk+alpha)/np.sum(ndk+alpha, axis=1, keepdims=True)
phi=(nkw+beta)/np.sum(nkw+beta, axis=1, keepdims=True)

plt.figure(figsize=(10, 4))
plt.plot(trace_topic0, label='Topic 0 proportion')
plt.title("Trace Plot: Topic 0 Proportion over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Proportion")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(trace_loglike, color='orange', label='Log-Likelihood')
plt.title("Convergence Diagnostic: Log-likelihood")
plt.xlabel("Iteration")
plt.ylabel("Log-Likelihood")
plt.legend()
plt.grid(True)
plt.show()
