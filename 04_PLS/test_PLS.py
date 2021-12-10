import numpy as np
import pyreadr

import sklearn
import sklearn.cross_decomposition
from sklearn.metrics import mean_squared_error as mse

from matplotlib import pyplot as plt

# Read in dataset
meats = pyreadr.read_r('meats.RData')
meats = np.array(meats['meats'])

meats_train = meats[:130]
meats_val = meats[130:175]
meats_test = meats[175:215]

X_train = meats_train[:, :-3]
Y_train = meats_train[:, -3:]

X_val = meats_val[:, :-3]
Y_val = meats_val[:, -3:]

X_test = meats_test[:, :-3]
Y_test = meats_test[:, -3:]

water_plot = np.zeros(100)
fat_plot = np.zeros(100)
protein_plot = np.zeros(100)
for n_comp in range(1, 101):
    my_plsr = sklearn.cross_decomposition.PLSRegression(n_components=n_comp, scale=True)
    my_plsr.fit(X_train, Y_train)
    preds = my_plsr.predict(X_val)

    water_rmse = np.sqrt(mse(Y_val[:, 0], preds[:, 0]))
    fat_rmse = np.sqrt(mse(Y_val[:, 1], preds[:, 1]))
    protein_rmse = np.sqrt(mse(Y_val[:, 2], preds[:, 2]))

    water_plot[n_comp-1] = water_rmse
    fat_plot[n_comp-1] = fat_rmse
    protein_plot[n_comp-1] = protein_rmse

fig, axs = plt.subplots(1,3, sharey=True, figsize=(8, 4))
comps = np.arange(1, 101)
axs[0].plot(comps, water_plot)
axs[0].set_title('Water')
axs[0].set_ylabel('RMSE')
axs[0].grid()

axs[1].plot(comps, fat_plot)
axs[1].set_title('Fat')
axs[1].set_xlabel('Components')
axs[1].grid()

axs[2].plot(comps, protein_plot)
axs[2].set_title('Protein')
axs[2].grid()

fig.savefig('PLS_meats.png', dpi=600)

plt.show()
