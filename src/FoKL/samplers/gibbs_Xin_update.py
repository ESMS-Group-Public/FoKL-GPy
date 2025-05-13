import numpy as np
import math
from numpy import linalg as LA
from scipy.linalg import eigh


class Sampler2:
    def __init__(self, fokl, config, functions):
        self.fokl = fokl
        self.config = config
        self.functions = functions

        
    def gibbs_Xin_update(self, sigsqd0, inputs, data, phis, Xin, discmtx, a, b, atau, btau, phind, xsm,
                                 mu_old, Sigma_old, draws):
        """
        This version of the sampler increases efficiency by accepting an set of
        inputs 'Xin' (matrix of data evaluated at basis function combinations)
        derived from earlier iterations.

        'sigsqd0' is the initial guess for the obs error variance

        'inputs' is the set of normalized inputs -- both parameters and model
        inputs -- with columns corresponding to inputs and rows the different
        experimental designs

        'data' are the experimental results: column vector, with entries
        corresponding to rows of 'inputs'

        'phis' are a data structure with the spline coefficients for the basis
        functions, built with 'BasisSpline.txt' and 'splineloader' or
        'splineconvert'

        'discmtx' is the interaction matrix for the bss-anova function -- rows
        are terms in the function and columns are inputs (cols should line up
        with cols in 'inputs'

        'a' and 'b' are the parameters of the ig distribution for the
        observation error variance of the data

        'atau' and 'btau' are the parameters of the ig distribution for the 'tau
        squared' parameter: the variance of the beta priors

        'mu_old' and 'Sigma_old' are the means and covariance matrix respectively
        of the priors of the previous model being updated.

        'draws' is the total number of draws
        """

        # building the matrix by calculating the corresponding basis function outputs for each set of inputs
        minp, ninp = np.shape(inputs)
        phi_vec = []
        if np.shape(discmtx) == ():  # part of fix for single input model
            mmtx = 1
        else:
            mmtx, null = np.shape(discmtx)

        if np.size(Xin) == 0:
            Xin = np.ones((minp, 1))
            mxin, nxin = np.shape(Xin)
        else:
            # X = Xin
            mxin, nxin = np.shape(Xin)
        if mmtx - nxin < 0:
            X = Xin
        else:
            X = np.append(Xin, np.zeros((minp, mmtx - nxin)), axis=1)

        for i in range(minp):  # for datapoint in training datapoints

            # ------------------------------
            # [IN DEVELOPMENT] PRINT PERCENT COMPLETION TO CONSOLE (reported to cause significant delay):
            #
            # if self.ConsoleOutput and data.dtype != np.float64:  # if large dataset, show progress for sanity check
            #     percent = i / (minp - 1)
            #     sys.stdout.write(f"Gibbs: {round(100 * percent, 2):.2f}%")  # show percent of data looped through
            #     sys.stdout.write('\r')  # set cursor at beginning of console output line (such that next iteration
            #         # of Gibbs progress (or [ind, ev] if at end) overwrites current Gibbs progress)
            #     sys.stdout.flush()
            #
            # [END]
            # ----------------------------

            for j in range(nxin, mmtx + 1):
                null, nxin2 = np.shape(X)
                if j == nxin2:
                    X = np.append(X, np.zeros((minp, 1)), axis=1)

                phi = 1

                for k in range(ninp):  # for input var in input vars

                    if np.shape(discmtx) == ():
                        num = discmtx
                    else:
                        num = discmtx[j - 1][k]

                    if num != 0:  # enter if loop if num is nonzero
                        nid = int(num - 1)

                        # Evaluate basis function:
                        if self.config.KERNELS[0] == self.config.DEFAULT['kernel']:  # == 'Cubic Splines':
                            coeffs = [phis[nid][order][phind[i, k]] for order in range(4)]  # coefficients for cubic
                        elif self.config.KERNELS[1] == self.config.DEFAULT['kernel']:  # == 'Bernoulli Polynomials':
                            coeffs = phis[nid]  # coefficients for bernoulli
                        phi = phi * self.functions.evaluate_basis(coeffs, xsm[i, k])  # multiplies phi(x0)*phi(x1)*etc.

                X[i][j] = phi

        # Once X is evaluated at the new data points, three different conditions are evaluated to
        # determine the appropriate methodology to pursue.  They are:
        # 1) No given mean or covariance strong prior (first time model)
        # 2) Re-evaluate model with strong prior, but same number of terms
        # 3) Create new terms for model given a strong prior

        # Case 1) New Model
        if np.size(mu_old) == 0:
            # initialize tausqd at the mode of its prior: the inverse of the mode of
            # sigma squared, such that the initial variance for the betas is 1
            # Start both at the mode of the prior


            tausqd = 1 / sigsqd0

            XtX = np.transpose(X).dot(X)

            Xty = np.transpose(X).dot(data)

            # See the link https://stackoverflow.com/questions/8765310/scipy-linalg-eig-return-complex-eigenvalues-for-covariance-matrix
            Lamb, Q = eigh(
                XtX)  # using scipy eigh function to avoid generation of imaginary values due to numerical errors
            # Lamb, Q = LA.eig(XtX)

            Lamb_inv = np.diag(1 / Lamb)

            betahat = Q.dot(Lamb_inv).dot(np.transpose(Q)).dot(Xty)
            # This is sum squared error, not just squared error or L2 Norm
            squerr = LA.norm(data - X.dot(betahat)) ** 2

            astar = a + 1 + len(data) / 2 + (mmtx + 1) / 2
            atau_star = atau + mmtx / 2

            dtd = np.transpose(data).dot(data)

            # Gibbs iterations

            betas = np.zeros((draws, mmtx + 1))
            sigs = np.zeros((draws, 1))
            taus = np.zeros((draws, 1))
            sigsqd = sigsqd0

            lik = np.zeros((draws, 1))
            n = len(data)

            for k in range(draws):
                # Step 1: Hold sigma squared and tau squared constant
                # This is to get the inverse of the new covariance
                Lamb_tausqd = np.diag(Lamb) + (1 / tausqd) * np.identity(mmtx + 1)
                Lamb_tausqd_inv = np.diag(1 / np.diag(Lamb_tausqd))

                # What does mun (mu new)
                mun = Q.dot(Lamb_tausqd_inv).dot(np.transpose(Q)).dot(Xty)
                S = Q.dot(np.diag(np.diag(Lamb_tausqd_inv) ** (1 / 2)))

                vec = np.random.normal(loc=0, scale=1, size=(mmtx + 1, 1))  # drawing from normal distribution
                betas[k][:] = np.transpose(mun + sigsqd ** (1 / 2) * (S).dot(vec))

                # Components for the likelihood
                comp1 = -(n / 2) * np.log(sigsqd)
                comp2 = np.transpose(betahat) - betas[k][:]
                comp3 = betahat - np.reshape(betas[k][:], (len(betas[k][:]), 1))
                # forcing array into a column for comp3, np.transpose not effective
                lik[k] = comp1 - (squerr + comp2.dot(XtX).dot(comp3)) / (2 * sigsqd)

                # vecc is difference between mu new and betas calculated
                vecc = mun - np.reshape(betas[k][:], (len(betas[k][:]), 1))

                # Step 2: Hold Beta and tau squared constant, calculate sigma squared
                comp1 = 0.5 * np.transpose(vecc)
                comp2 = (XtX + (1 / tausqd) * np.identity(mmtx + 1)).dot(vecc)
                comp3 = 0.5 * np.transpose(mun).dot(Xty)

                bstar = b + comp1.dot(comp2) + 0.5 * dtd - comp3

                # Returning a 'not a number' constant if bstar is negative, which would
                # cause np.random.gamma to return a ValueError
                if bstar < 0:
                    sigsqd = math.nan
                else:
                    sigsqd = 1 / np.random.gamma(astar, 1 / bstar)

                sigs[k] = sigsqd

                # Step 3: Hold Beta and sigma squared constant, calculate tau squared
                btau_star = (1 / (2 * sigsqd)) * (
                    betas[k][:].dot(np.reshape(betas[k][:], (len(betas[k][:]), 1)))) + btau

                tausqd = 1 / np.random.gamma(atau_star, 1 / btau_star)
                taus[k] = tausqd

            # Calculate the evidence (BIC)
            ev = (mmtx + 1) * np.log(n) - 2 * max(lik)

            X = X[:, 0:mmtx]

            return betas, sigs, taus, X, ev

        # Case 2) Given Strong Prior (Old Model), no new terms
        elif np.shape(mu_old)[1] == mmtx + 1:
            print('same')
            # seperate into a section with X_old and X_new based on the existing
            # parameters noted in Xin

            null, num_old_terms = np.shape(mu_old)

            length_old = num_old_terms

            X_old = X

            # initialize tausqd at the mode of its prior: the inverse of the mode of
            # sigma squared, such that the initial variance for the betas is 1
            # Start both at the mode of the prior
            tausqd = 1 / sigsqd0

            # Precompute Terms related to 'old' values
            XotXo = np.asmatrix(np.transpose(X_old).dot(X_old))

            Xoty = np.transpose(X_old).dot(data)

            Sigma_old_inverse = np.linalg.inv(Sigma_old)

            # Precompute terms for sigma squared and tau squared
            astar = a + len(data) / 2 + (mmtx + 1) / 2
            atau_star = atau + (mmtx + 1) / 2

            yty = np.transpose(data).dot(data)
            ytXo = np.transpose(data).dot(X_old)

            # Gibbs iterations

            betas_old = np.asmatrix(np.zeros((draws, num_old_terms)))
            sigs = np.zeros((draws, 1))
            taus = np.zeros((draws, 1))
            sigsqd = sigsqd0

            lik = np.zeros((draws, 1))
            n = len(data)

            mu_old = mu_old.transpose()

            for k in range(draws):
                # Step 1: Hold betas_new, sigma squared, and tau squared constant. Get betas_old
                Sigma_old_inverse_post = XotXo + (1 / tausqd) * Sigma_old_inverse
                Sigma_old_post = np.linalg.inv(Sigma_old_inverse_post)

                # See the link https://stackoverflow.com/questions/8765310/scipy-linalg-eig-return-complex-eigenvalues-for-covariance-matrix
                Lamb_old, Q_old = eigh(XotXo + (
                            1 / tausqd) * Sigma_old_inverse)  # using scipy eigh function to avoid generation of imaginary values due to numerical errors
                # Lamb_tausqd_inv_old = np.diag(np.linalg.inv((np.diag(Lamb_old))))
                Lamb_tausqd_inv_old = 1/Lamb_old

                # Mean of distribution
                mu_old_first_part = (Xoty + (1 / tausqd * Sigma_old_inverse).dot(mu_old))
                mu_old_post = Sigma_old_post.dot(mu_old_first_part)
                # print('Sigma_old_inverse.dot(mu_old)', Sigma_old_inverse.dot(mu_old))
                S_old = Q_old.dot((np.diag(Lamb_tausqd_inv_old) ** (1 / 2)))

                # Random draw for sampling
                vec_old = np.random.normal(loc=0, scale=1, size=(length_old, 1))  # drawing from normal distribution
                betas_old[k][:] = np.transpose(mu_old_post + sigsqd ** (1 / 2) * (S_old).dot(vec_old))

                # Step 2: Hold betas_old, betas_new, and tau squared constant, calculate sigma squared
                comp1 = 0.5 * (yty - ytXo.dot(betas_old[k][:].transpose()))
                comp2 = 0.5 * (-(betas_old[k][:]).dot(Xoty) + (betas_old[k][:]).dot(XotXo).dot(
                    betas_old[k][:].transpose()))
                comp3 = 0.5 * (1 / tausqd) * (
                            (betas_old[k][:]).dot(Sigma_old_inverse).dot(betas_old[k][:].transpose()) - (
                    betas_old[k][:]).dot(Sigma_old_inverse).dot(mu_old))
                comp4 = 0.5 * (1 / tausqd) * (-np.transpose(mu_old).dot(Sigma_old_inverse).dot(
                    betas_old[k][:].transpose()) + np.transpose(mu_old).dot(Sigma_old_inverse).dot(mu_old))

                bstar = comp1 + comp2 + comp3 + comp4 + b
                # Returning a 'not a number' constant if bstar is negative, which would
                # cause np.random.gamma to return a ValueError
                if bstar < 0:
                    sigsqd = math.nan
                else:
                    sigsqd = 1 / np.random.gamma(astar, 1 / bstar)

                sigs[k] = sigsqd

                # Step 3: Hold betas_old, betas_new, and sigma squared constant, calculate tau squared
                comp1 = 0.5 * (1 / sigsqd) * (
                            (betas_old[k][:]).dot(Sigma_old_inverse).dot(betas_old[k][:].transpose()) - (
                    betas_old[k][:]).dot(Sigma_old_inverse).dot(mu_old))
                comp2 = 0.5 * (1 / sigsqd) * (-np.transpose(mu_old).dot(Sigma_old_inverse).dot(
                    betas_old[k][:].transpose()) + np.transpose(mu_old).dot(Sigma_old_inverse).dot(mu_old))

                btau_star = comp1 + comp2 + btau

                tausqd = 1 / np.random.gamma(atau_star, 1 / btau_star)
                taus[k] = tausqd

                # Step 5: for BIC, calculate the Natural Log Likelihood
                comp1 = -(n / 2) * np.log(sigsqd)
                comp2 = yty - ytXo.dot(betas_old[k][:].transpose())
                comp3 = -(betas_old[k][:]).dot(Xoty) + (betas_old[k][:]).dot(XotXo).dot(betas_old[k][:].transpose())

                lik[k] = comp1 - 0.5 / sigsqd * (comp2 + comp3)

            # Calculate the evidence (BIC)
            ev = (mmtx + 1) * np.log(n) - 2 * max(lik)

            # Updated X matrix, not differentiated between old and new though
            X = X[:, 0:mmtx]

            betas = betas_old

            return betas, sigs, taus, X, ev

        # Case 3) Give Strong Prior (Old Model), creating new terms
        elif np.shape(mu_old)[1] < mmtx + 1:
            print('new')
            # seperate into a section with X_old and X_new based on the existing
            # parameters noted in Xin

            null, num_old_terms = np.shape(mu_old)

            length_old = num_old_terms
            length_new = mmtx - num_old_terms + 1

            X_old = X[:, 0:length_old]
            X_new = X[:, length_old: length_old + length_new]

            # initialize tausqd at the mode of its prior: the inverse of the mode of
            # sigma squared, such that the initial variance for the betas is 1
            # Start both at the mode of the prior
            tausqd = 1 / sigsqd0

            # Precompute Terms related to 'old' values
            XotXo = np.asmatrix(np.transpose(X_old).dot(X_old))

            Xoty = np.transpose(X_old).dot(data)

            Sigma_old_inverse = np.linalg.inv(Sigma_old)
            Sigma_old_inverse_post = XotXo + Sigma_old_inverse
            Sigma_old_post = np.linalg.inv(Sigma_old_inverse_post)

            # See the link https://stackoverflow.com/questions/8765310/scipy-linalg-eig-return-complex-eigenvalues-for-covariance-matrix
            Lamb_old, Q_old = eigh(
                XotXo + Sigma_old_inverse)  # using scipy eigh function to avoid generation of imaginary values due to numerical errors
            # Lamb, Q = LA.eig(XtX)

            # Lamb_old_inv = np.diag(1/Lamb_old)

            # betahat_old = Q_old.dot(Lamb_old_inv).dot(np.transpose(Q_old)).dot(Xoty)
            # This is sum squared error, not just squared error or L2 Norm
            # squerr_old = LA.norm(data - X_old.dot(betahat_old)) ** 2

            # Precompute Terms related to 'new' values
            XntXn = np.transpose(X_new).dot(X_new)

            Xnty = np.transpose(X_new).dot(data)

            # See the link https://stackoverflow.com/questions/8765310/scipy-linalg-eig-return-complex-eigenvalues-for-covariance-matrix
            Lamb_new, Q_new = eigh(
                XntXn)  # using scipy eigh function to avoid generation of imaginary values due to numerical errors
            # Lamb, Q = LA.eig(XtX)

            # Lamb_new_inv = np.diag(np.linalg.inv(Lamb_new))

            # betahat_new = Q_new.dot(Lamb_new_inv).dot(np.transpose(Q_new)).dot(Xnty)
            # This is sum squared error, not just squared error or L2 Norm
            # squerr_new = LA.norm(data - X_new.dot(betahat_new)) ** 2

            XotXn = np.transpose(X_old).dot(X_new)
            XntXo = np.transpose(X_new).dot(X_old)

            # Sigma_old_inverse = np.linalg.inv(Sigma_old)
            # Sigma_old_inverse_post = XotXo+Sigma_old_inverse
            # Sigma_old_post = np.linalg.inv(Sigma_old_inverse_post)
            # print('Sigma Old Post', Sigma_old_post)

            # Lamb_tausqd_old = np.diag(Lamb_old) + Sigma_old
            Lamb_tausqd_inv_old = np.diag(np.linalg.inv((np.diag(Lamb_old))))
            # print('Lamb_tausqd_old', np.shape(Lamb_tausqd_old))
            # print('Lamb_tausqd_inv_old', np.shape(Lamb_tausqd_inv_old))
            # print('Q_old', np.shape(Q_old))

            # Precompute terms for sigma squared and tau squared
            astar = a + len(data) / 2 + (mmtx + 1) / 2
            atau_star = atau + (length_new) / 2

            yty = np.transpose(data).dot(data)
            ytXo = np.transpose(data).dot(X_old)
            ytXn = np.transpose(data).dot(X_new)

            # Gibbs iterations

            betas_old = np.asmatrix(np.zeros((draws, num_old_terms)))
            betas_new = np.asmatrix(np.zeros((draws, mmtx - num_old_terms + 1)))
            sigs = np.zeros((draws, 1))
            taus = np.zeros((draws, 1))
            sigsqd = sigsqd0

            lik = np.zeros((draws, 1))
            n = len(data)

            mu_old = mu_old.transpose()

            for k in range(draws):
                # Step 1: Hold betas_new, sigma squared, and tau squared constant. Get betas_old

                # Mean of distribution
                mu_old_first_part = Xoty - XotXn.dot(betas_new[k - 1].transpose()) + Sigma_old_inverse.dot(mu_old)
                mu_old_post = Sigma_old_post.dot(mu_old_first_part)
                S_old = Q_old.dot((np.diag(Lamb_tausqd_inv_old) ** (1 / 2)))

                vec_old = np.random.normal(loc=0, scale=1, size=(length_old, 1))  # drawing from normal distribution
                betas_old[k][:] = np.transpose(mu_old_post + sigsqd ** (1 / 2) * (S_old).dot(vec_old))

                # Step 2: Hold betas_new, sigma squared, and tau squared constant. Get betas_old
                # This is to get the inverse of the new covariance
                Lamb_tausqd_new = np.diag(Lamb_new) + (1 / tausqd) * np.identity(length_new)
                Lamb_tausqd_inv_new = np.diag(np.linalg.inv(Lamb_tausqd_new))

                # Mean of distribution
                mu_new_first_part = Xnty - XntXo.dot(betas_old[k].transpose())
                Sigma_new_inverse_post = XntXn + (1 / tausqd) * np.identity(length_new)
                mu_new_post = (np.linalg.inv(Sigma_new_inverse_post)).dot(mu_new_first_part)
                S_new = Q_new.dot((np.diag(Lamb_tausqd_inv_new) ** (1 / 2)))

                vec_new = np.random.normal(loc=0, scale=1, size=(length_new, 1))  # drawing from normal distribution
                betas_new[k][:] = np.transpose(mu_new_post + sigsqd ** (1 / 2) * (S_new).dot(vec_new))

                # Step 3: Hold betas_old, betas_new, and tau squared constant, calculate sigma squared
                comp1 = 0.5 * (yty - ytXo.dot(betas_old[k][:].transpose()) - ytXn.dot(betas_new[k][:].transpose()))
                comp2 = 0.5 * (-(betas_old[k][:]).dot(Xoty) + (betas_old[k][:]).dot(XotXo).dot(
                    betas_old[k][:].transpose()) + (betas_old[k][:]).dot(XotXn).dot(betas_new[k][:].transpose()))
                comp3 = 0.5 * (-(betas_new[k][:]).dot(Xnty) + (betas_new[k][:]).dot(XntXo).dot(
                    betas_old[k][:].transpose()) + (betas_new[k][:]).dot(XntXn).dot(betas_new[k][:].transpose()))
                comp4 = 0.5 / tausqd * ((betas_new[k][:]).dot(betas_new[k][:].transpose()))
                comp5 = 0.5 * ((betas_old[k][:]).dot(Sigma_old_inverse).dot(betas_old[k][:].transpose()) - (
                betas_old[k][:]).dot(Sigma_old_inverse).dot(mu_old))
                comp6 = 0.5 * (-np.transpose(mu_old).dot(Sigma_old_inverse).dot(
                    betas_old[k][:].transpose()) + np.transpose(mu_old).dot(Sigma_old_inverse).dot(mu_old))

                bstar = comp1 + comp2 + comp3 + comp4 + comp5 + comp6 + b
                # Returning a 'not a number' constant if bstar is negative, which would
                # cause np.random.gamma to return a ValueError
                if bstar < 0:
                    sigsqd = math.nan
                else:
                    sigsqd = 1 / np.random.gamma(astar, 1 / bstar)

                sigs[k] = sigsqd

                # Step 4: Hold betas_old, betas_new, and sigma squared constant, calculate tau squared
                btau_star = (1 / (2 * sigsqd)) * (betas_new[k][:].dot(betas_new[k][:].transpose())) + btau

                tausqd = 1 / np.random.gamma(atau_star, 1 / btau_star)
                taus[k] = tausqd

                # Step 5: for BIC, calculate the Natural Log Likelihood
                comp1 = -(n / 2) * np.log(sigsqd)
                comp2 = yty - ytXo.dot(betas_old[k][:].transpose()) - ytXn.dot(betas_new[k][:].transpose())
                comp3 = -(betas_old[k][:]).dot(Xoty) + (betas_old[k][:]).dot(XotXo).dot(
                    betas_old[k][:].transpose()) + (betas_old[k][:]).dot(XotXn).dot(betas_new[k][:].transpose())
                comp4 = -(betas_new[k][:]).dot(Xnty) + (betas_new[k][:]).dot(XntXo).dot(
                    betas_old[k][:].transpose()) + (betas_new[k][:]).dot(XntXn).dot(betas_new[k][:].transpose())

                lik[k] = comp1 - 0.5 / sigsqd * (comp2 + comp3 + comp4)

            # Calculate the evidence (BIC)
            ev = (mmtx + 1) * np.log(n) - 2 * max(lik)

            # Updated X matrix, not differentiated between old and new though
            X = X[:, 0:mmtx]

            betas = np.concatenate((betas_old, betas_new), axis=1)

            return betas, sigs, taus, X, ev


        # Case 4) Something unexpected happens
        else:
            print('Error: No appropriate cases for evaluation found.')
            return