import numpy as np
import statsmodels.api as sm

np.random.seed(0)

hellwig = [0, 0, 0]
aic = [0, 0, 0]
bic = [0, 0, 0]

n = 1000

for _ in range(100000):
    z_1 = [np.random.normal(0, 1) for _ in range(n)]
    z_2 = [np.random.normal(0, 1) for _ in range(n)]

    x_1 = [el_1 for el_1 in z_1]
    x_2 = [1/4 * el_1 + np.sqrt(15)/4 * el_2
           for (el_1, el_2) in zip(z_1, z_2)]
    y = [4 * el_1 + el_2
         for (el_1, el_2) in zip(x_1, x_2)]

    hellwig_s_x_1 = (np.corrcoef(x_1, y)[0, 1])**2
    hellwig_s_x_2 = (np.corrcoef(x_2, y)[0, 1])**2
    hellwig_s_x_1_x_2 = (((np.corrcoef(x_1, y)[0, 1])**2 +
                          (np.corrcoef(x_2, y)[0, 1])**2) /
                         (1 + abs(np.corrcoef(x_1, x_2)[0, 1])))
    max_hellwig = max(hellwig_s_x_1, hellwig_s_x_2, hellwig_s_x_1_x_2)
    for i in range(3):
        if [hellwig_s_x_1, hellwig_s_x_2,
            hellwig_s_x_1_x_2][i] == max_hellwig:
            hellwig[i] = hellwig[i] + 1

    model_1 = sm.OLS(y, sm.add_constant(x_1)).fit()
    model_2 = sm.OLS(y, sm.add_constant(x_2)).fit()
    model_3 = sm.OLS(y, sm.add_constant(
        np.column_stack((x_1, x_2)))).fit()

    aic_s_x_1 = model_1.aic
    aic_s_x_2 = model_2.aic
    aic_s_x_1_x_2 = model_3.aic
    min_aic = min(aic_s_x_1, aic_s_x_2, aic_s_x_1_x_2)
    for i in range(3):
        if [aic_s_x_1, aic_s_x_2,
            aic_s_x_1_x_2][i] == min_aic:
            aic[i] = aic[i] + 1

    bic_s_x_1 = model_1.bic
    bic_s_x_2 = model_2.bic
    bic_s_x_1_x_2 = model_3.bic
    min_bic = min(bic_s_x_1, bic_s_x_2, bic_s_x_1_x_2)
    for i in range(3):
        if [bic_s_x_1, bic_s_x_2,
            bic_s_x_1_x_2][i] == min_bic:
            bic[i] = bic[i] + 1

print(f'{[el/100000 for el in hellwig]} \n'
      f'{[el/100000 for el in aic]} \n'
      f'{[el/100000 for el in bic]} \n')