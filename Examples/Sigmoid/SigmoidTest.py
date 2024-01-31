from FoKL import FoKLRoutines
import numpy as np


def main():

    # Inputs:
    X_grid = np.loadtxt('X.csv',dtype=float,delimiter=',')
    Y_grid = np.loadtxt('Y.csv',dtype=float,delimiter=',')

    # Data:
    Z_grid = np.loadtxt('DATA_nois.csv',dtype=float,delimiter=',')

    # Reshaping grid matrices into vectors via fortran index order:
    m, n = np.shape(X_grid) # = np.shape(Y_grid) = np.shape(Z_grid) = dimensions of grid
    X = np.reshape(X_grid, (m*n,1), order='F')
    Y = np.reshape(Y_grid, (m*n,1), order='F')
    Z = np.reshape(Z_grid, (m*n,1), order='F')

    # Initializing FoKL model with some user-defined hyperparameters (leaving others blank for default values):
    model = FoKLRoutines.FoKL(a=9, b=0.01, atau=3, btau=4000, aic=True)

    # Training FoKL model on a random selection of 100% (or 100%, 80%, 60%, etc.) of the dataset:
    train_all = [1] # = [1, 0.8, 0.6] etc. if sweeping through the percentage of data to train on
    betas_all = []
    mtx_all = []
    evs_all = []
    meen_all = []
    bounds_all = []
    rmse_all = []
    for train in train_all:

        print("\nCurrently fitting model to",train * 100,"% of data ...")

        # Running emulator routine to fit model to training data as a function of the corresponding training inputs:
        betas, mtx, evs = model.fit([X, Y], Z, train=train)

        # Provide feedback to user before the figure from coverage3() pops up and pauses the code:
        print("\nDone! Please close the figure to continue.\n")

        # Evaluating and visualizing predicted values of data as a function of all inputs (train set plus test set):
        title = 'FoKL Model Trained on ' + str(train * 100) + '% of Data'
        meen, bounds, rmse = model.coverage3(plot='bounds',title=title,legend=1)

        # Store any values from iteration if performing additional post-processing or analysis:
        betas_all.append(betas)
        mtx_all.append(mtx)
        evs_all.append(evs)
        meen_all.append(meen)
        bounds_all.append(bounds)
        rmse_all.append(rmse)

        # Reset the model so that all attributes of the FoKL class are removed except for the hyperparameters:
        model.clear()

    # Post-processing:
    print("\nThe results are as follows:")
    for ii in range(len(train_all)):
        print("\n   ",train_all[ii]*100,"% of Data:\n    --> RMSE =",rmse_all[ii])


if __name__ == '__main__':
    main()

