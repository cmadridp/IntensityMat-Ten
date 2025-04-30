from generate import *

from kde import kernel_density

from tensor_prediction_stage import tensor_prediction


dim=8
N_train= 5000
##############tuning parameter selection
MM=20
if N_train<2**dim*MM:
    print('insufficient data')
    LL =1
else:
    LL=2
#LL=2
tensor_shape=[LL for _ in range(dim)]
tensor_shape[0]=MM
print('M=', MM, 'L=',  LL)



###S1: Gaussian_mixture
#########################################

lr_error=0
kde_error=0


# Define parameters for the Gaussian mixture
mean = [0.3, 0.7]
cov = [0.1, 0.1]
scale= 10
# Create a Gaussian mixture instance
gmm = gaussian_mixture(dim, mean, cov)
# Create a Poisson point process instance with intensity proportional to the Gaussian mixture density.
# Here, scale controls the overall expected number of points.
ppp = poisson_pp(gmm, dim, scale, region_lower=[-0.5]*dim, region_upper=[0.5]*dim) # Simulate the Poisson point process.

for rr in range(3):
    point_processes= ppp.generate(N_train)
    X_train = np.concatenate(point_processes, axis=0)
    X_train_domain = domain(X_train)
    X_train_0_1 = X_train_domain.transform_to_0_1(X_train)

    lr= tensor_prediction( tensor_shape, dim, MM , X_train_0_1)

    N_test = 10000
    X_new = np.random.uniform(0,1, N_test*dim).reshape((N_test, dim))

    y_lr= X_train_domain.transform_density_val(np.array([lr.predict(xx) for xx in X_new])) * X_train.shape[0] / N_train

    y_true = gmm.density_value(X_train_domain.transform_from_0_1(X_new)) * scale

    lr_error+=np.linalg.norm(y_lr - y_true,2)**2/np.linalg.norm(y_true,2)**2

    y_kde=kernel_density().compute(dim, X_train, X_train_domain.transform_from_0_1(X_new)) * X_train.shape[0] / N_train
    kde_error+=np.linalg.norm(y_kde - y_true,2)**2/np.linalg.norm(y_true,2)**2
    print("lr errors =",  lr_error/(rr+1), "kde errors =", kde_error/(rr+1))




###S2: SinusoidalDensity
#########################################

lr_error=0
kde_error=0



amplitude= 10.0
scale= 10
# Create a SinusoidalDensity
density = SinusoidalDensity(amplitude)
# Create a Poisson point process instance with intensity proportional to the SinusoidalDensity.
# Here, scale controls the overall expected number of points.
ppp = poisson_pp(density, dim, scale, region_lower=[-0.5]*dim, region_upper=[0.5]*dim)  # Simulate the Poisson point process.

for rr in range(3):
    point_processes= ppp.generate(N_train)
    X_train = np.concatenate(point_processes, axis=0)
    X_train_domain = domain(X_train)
    X_train_0_1 = X_train_domain.transform_to_0_1(X_train)

    lr= tensor_prediction( tensor_shape, dim, MM , X_train_0_1)

    N_test = 10000
    X_new = np.random.uniform(0,1, N_test*dim).reshape((N_test, dim))

    y_lr= X_train_domain.transform_density_val(np.array([lr.predict(xx) for xx in X_new])) * X_train.shape[0] / N_train

    y_true = density.density_value(X_train_domain.transform_from_0_1(X_new)) * scale

    lr_error+=np.linalg.norm(y_lr - y_true,2)**2/np.linalg.norm(y_true,2)**2

    y_kde=kernel_density().compute(dim, X_train, X_train_domain.transform_from_0_1(X_new)) * X_train.shape[0] / N_train
    kde_error+=np.linalg.norm(y_kde - y_true,2)**2/np.linalg.norm(y_true,2)**2
    print("lr errors =",  lr_error/(rr+1), "kde errors =", kde_error/(rr+1))
