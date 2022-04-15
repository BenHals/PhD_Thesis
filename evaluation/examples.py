#%%
import matplotlib.pyplot as plt
import numpy as np
from PhDCode.utils.eval_utils import set_plot_style

set_plot_style()


#%%
def noise_example_labelling_function(x):
    x1, x2 = x
    return x1 * x2
#%%
#RANDOM CONTEXT
x1 = np.random.randint(0, 2, 600)
x2 = np.random.randint(0, 2, 600)
y = np.array(list(map(noise_example_labelling_function, zip(x1, x2))))

fig, ax = plt.subplots(figsize=(10, 5))
cm = 1/2.54
plt.subplots(figsize=(8*cm, 5*cm))
plt.plot(list(range(x1.shape[0])), x2, lw=1)
plt.ylabel("Context")
plt.xlabel("$t$")
plt.savefig("noise_H_example.pdf", bbox_inches='tight')
# plt.show()

#%%
#RANDOM CONTEXT
x1 = np.random.randint(0, 2, 600)
x2 = np.array([*[0]*100, *[1]*100]*3)
y = np.array(list(map(noise_example_labelling_function, zip(x1, x2))))

fig, ax = plt.subplots(figsize=(10, 5))
plt.plot(list(range(x1.shape[0])), x2, lw=1)
plt.ylabel("Context")
plt.xlabel("t")
plt.savefig("context_H_example.pdf")
