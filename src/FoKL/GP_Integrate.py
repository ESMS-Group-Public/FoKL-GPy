import numpy as np

"""""
betas is a list of arrays in which each entry to the list contains a specific row of the betas matrix, 
or the mean of the the betas matrix for each model being integrated
matrix is a list of arrays containing the interaction matrix of each model
b is an array of of the values of all the other inputs to the model(s) (including
any forcing functions) over the time period we integrate over. The length of b 
should be equal to the number of points in the final time series (end-start)/h
All values in b need to be normalized with respect to the min and max values
of their respective values in the training dataset
h is the step size with respect to time
norms is a matrix of the min and max values of all the inputs being 
integrated (in the same order as y0). min values are in the top row, max values in the bottom.
Start is the time at which integration begins. Stop is the time to
end integration.
y0 is an array of the inital conditions for the models being integrated
Used inputs is a list of arrays containing the information as to what inputs
are used in what model. Each array should contain a vector corresponding to a different model. 
Inputs should be referred to as those being integrated first, followed by
those contained in b (in the same order as they appear in y0 and b
respectively)
For example, if two models were being integrated, with 3 other inputs total
and the 1st model used both models outputs as inputs and the 1st and 3rd additional
inputs, while the 2nd model used its own output as an input and the 2nd
and 3rd additional inputs, used_inputs would be equal to
[[1,1,1,0,1],[0,1,0,1,0]]. 
If the models created do not follow this ordering scheme for their inputs
the inputs can be rearranged based upon an alternate 
numbering scheme provided to used_inputs. E.g. if the inputs need to be
reordered the the 1st input should have a '1' in its place in the
used_inputs vector, the 2nd input should have a '2' and so on. Using the
same example as before, if the 1st models inputs needed rearranged so that
the 3rd additional input came first, followed by the two model outputs in
the same order as they are in y0, and ends with the 1st additional input,
then the 1st cell in used_inputs would have the form [2,3,4,0,1]
T an array of the time steps the models are integrated at.
Y is an array of the models that have been integrated, at the time steps
contained in T.
"""


def GP_Integrate(betas, matrix, b, norms, phis, start, stop, y0, h, used_inputs):
    """""
      betas is a list of arrays in which each entry to the list contains a specific row of the betas matrix,
      or the mean of the the betas matrix for each model being integrated

      matrix is a list of arrays containing the interaction matrix of each model

      b is an array of of the values of all the other inputs to the model(s) (including
      any forcing functions) over the time period we integrate over. The length of b
      should be equal to the number of points in the final time series (end-start)/h
      All values in b need to be normalized with respect to the min and max values
      of their respective values in the training dataset

      h is the step size with respect to time

      norms is a matrix of the min and max values of all the inputs being
      integrated (in the same order as y0). min values are in the top row, max values in the bottom.

      Start is the time at which integration begins. Stop is the time to
      end integration.

      y0 is an array of the inital conditions for the models being integrated

      Used inputs is a list of arrays containing the information as to what inputs
      are used in what model. Each array should contain a vector corresponding to a different model.
      Inputs should be referred to as those being integrated first, followed by
      those contained in b (in the same order as they appear in y0 and b
      respectively)
      For example, if two models were being integrated, with 3 other inputs total
      and the 1st model used both models outputs as inputs and the 1st and 3rd additional
      inputs, while the 2nd model used its own output as an input and the 2nd
      and 3rd additional inputs, used_inputs would be equal to
      [[1,1,1,0,1],[0,1,0,1,0]].
      If the models created do not follow this ordering scheme for their inputs
      the inputs can be rearranged based upon an alternate
      numbering scheme provided to used_inputs. E.g. if the inputs need to breordered the the 1st input should have a '1' in its place in the
      used_inputs vector, the 2nd input should have a '2' and so on. Using the
      same example as before, if the 1st models inputs needed rearranged so that
      the 3rd additional input came first, followed by the two model outputs in
      the same order as they are in y0, and ends with the 1st additional input,
      then the 1st cell in used_inputs would have the form [2,3,4,0,1]

      T an array of the time steps the models are integrated at.

      Y is an array of the models that have been integrated, at the time steps
      contained in T.
      """
    def prediction(inputs):
        f = []
        for kk in range(len(inputs)):
            if len(f) == 0:
                f = [FokL.bss_eval(inputs[kk], betas[kk], phis, matrix[kk]) + betas[kk][len(betas[kk]) - 1]]
            else:
                f = np.append(f, FokL.bss_eval(inputs[kk], betas[kk], phis, matrix[kk]) + betas[kk][len(betas[kk]) - 1])
        return f

    def reorder(used, inputs):
        order = used[used != 0]
        reinputs = np.array((inputs.shape))
        for i in range(len(inputs)):
            reinputs[order[i] - 1] = inputs[i]
        return reinputs

    def normalize(v, minim, maxim):
        norm = np.zeros((1, 1))
        norm[0] = (v - minim) / (maxim - minim)
        if norm[0] > 1:
            norm[0] = 1
        if norm[0] < 0:
            norm[0] = 0
        return norm

    def bss_eval(x, betas, phis, mtx, Xin=[]):
        """
        x are normalized inputs
        betas are coefficients. If using 'Xin' include all betas (include constant beta)

        phis are the spline coefficients for the basis functions (cell array)

        mtx is the 'interaction matrix' -- a matrix each row of which corresponds
        to a term in the expansion, each column corresponds to an input. if the
        column is zero there's no corresponding basis function in the term; if
        it's greater than zero it corresponds to the order of the basis function

        Xin is an optional input of the chi matrix. If this was pre-computed with xBuild,
        one may use it to improve performance.
        """

        if Xin == []:
            m, n = np.shape(mtx)  # getting dimensions of the matrix 'mtx'

            delta = 0

            phind = []
            for j in range(len(x)):
                phind.append(math.floor(x[j] * 499))

            phind_logic = []
            for k in range(len(phind)):
                if phind[k] == 499:
                    phind_logic.append(1)
                else:
                    phind_logic.append(0)

            phind = np.subtract(phind, phind_logic)

            r = 1 / 499
            xmin = r * np.array(phind)
            X = (x - xmin) / r

            for i in range(m):
                phi = 1

                for j in range(n):

                    num = mtx[i][j]

                    if num != 0:
                        phi = phi * (phis[int(num) - 1][0][phind[j]] + phis[int(num) - 1][1][phind[j]] * X[j] \
                                     + phis[int(num) - 1][2][phind[j]] * X[j] ** 2 + phis[int(num) - 1][3][phind[j]] *
                                     X[j] ** 3)

                delta = delta + betas[i] * phi
        else:

            if np.ndim(betas) == 1:
                betas = np.array([betas])
            elif np.ndim(betas) > 2:
                print("The \'betas\' parameter has %d dimensions, but needs to have only 2." % (np.ndim(betas)))
                print("The current shape is:", np.shape(betas))
                print("Attempting to get rid of unnecessary dimensions of size 1...")
                betas = np.squeeze(betas)

                if np.ndim(betas) == 1:
                    betas = np.array([betas])
                    print("Success! New shape is", np.shape(betas))
                elif np.ndim(betas) == 2:
                    print("Success! New shape is", np.shape(betas))

            delta = Xin.dot(betas.T)

        return delta

    T = np.arange(start, stop + h, h)
    y = (y0.astype(float)).reshape(5, 1)
    Y = np.array([y0])
    Y = Y.reshape(len(y0), 1)

    ind = 1
    for t in range(len(T) - 1):
        inputs1 = list()
        othinputs = list()
        inputs2 = list()
        inputs3 = list()
        inputs4 = list()
        for i in range(len(y)):  # initialize inputs1 and othinputs to contain empty arrays
            inputs1.append([])
            othinputs.append([])
            inputs2.append([])
            inputs3.append([])
            inputs4.append([])
        for i in range(len(y)):
            for j in range(len(y)):
                if used_inputs[i][j] != 0:
                    if len(inputs1[i]) == 0:
                        inputs1[i] = normalize(y[j], norms[0, j], norms[1, j])
                    else:
                        inputs1[i] = np.append(inputs1[i], normalize(y[j], norms[0, j], norms[1, j]), 1)
        # if b.shape[0] > 0:
        #     for ii in range(len(y0)):
        #         for jj in range(len(y),b.shape[1]+len(y)):
        #             print(range(len(y),b.shape[1]+len(y)))
        #             if used_inputs[ii][jj] != 0:
        #                 if len(othinputs[ii])==0:
        #                     othinputs[ii] = b[ind-1,jj-len(y0)]
        #                 else:
        #                     othinputs[ii] = np.append(othinputs[ii],b[ind-1,jj-len(y0)],1)
        #     for k in range(len(y)):
        #         inputs1[k] = np.append(inputs1[k], othinputs[k])
        for ii in range(len(y0)):
            if np.amax(used_inputs[ii]) > 1:
                inputs1[ii] = reorder(used_inputs[ii], inputs1[ii])

        dy1 = prediction(inputs1) * h
        for p in range(len(y)):
            if y[p] >= norms[1, p] and dy1[p] > 0:
                dy1[p] = 0
            else:
                if y[p] <= norms[0, p] and dy1[p] < 0:
                    dy1[p] = 0

        for i in range(len(y)):
            for j in range(len(y)):
                if used_inputs[i][j] != 0:
                    if len(inputs2[i]) == 0:
                        inputs2[i] = normalize(y[j] + dy1[j] / 2, norms[0, j], norms[1, j])
                    else:
                        inputs2[i] = np.append(inputs2[i], normalize(y[j] + dy1[j] / 2, norms[0, j], norms[1, j]),
                                               1)
        # if b.shape[1] > 0:
        #     for k in range(len(y)):
        #         inputs2[k] = np.append(inputs2[k], othinputs[k])
        for ii in range(len(y0)):
            if np.amax(used_inputs[ii]) > 1:
                inputs2[ii] = reorder(used_inputs[ii], inputs2[ii])
        dy2 = prediction(inputs2) * h
        for p in range(len(y)):
            if (y[p] + dy1[p] / 2) >= norms[1, p] and dy2[p] > 0:
                dy2[p] = 0
            if (y[p] + dy1[p] / 2) <= norms[0, p] and dy2[p] < 0:
                dy2[p] = 0

        for i in range(len(y)):
            for j in range(len(y)):
                if used_inputs[i][j] != 0:
                    if len(inputs3[i]) == 0:
                        inputs3[i] = normalize(y[j] + dy2[j] / 2, norms[0, j], norms[1, j])
                    else:
                        inputs3[i] = np.append(inputs3[i], normalize(y[j] + dy2[j] / 2, norms[0, j], norms[1, j]),
                                               1)
        # if b.shape[1] > 0:
        #     for k in range(len(y)):
        #         inputs3[k] = np.append(inputs3[k], othinputs[k])
        for ii in range(len(y0)):
            if np.amax(used_inputs[ii]) > 1:
                inputs3[ii] = reorder(used_inputs[ii], inputs3[ii])
        dy3 = prediction(inputs3) * h
        for p in range(len(y)):
            if (y[p] + dy2[p] / 2) >= norms[1, p] and dy3[p] > 0:
                dy3[p] = 0
            if (y[p] + dy2[p] / 2) <= norms[0, p] and dy3[p] < 0:
                dy3[p] = 0

        for i in range(len(y)):
            for j in range(len(y)):
                if used_inputs[i][j] != 0:
                    if len(inputs4[i]) == 0:
                        inputs4[i] = normalize(y[j] + dy3[j], norms[0, j], norms[1, j])
                    else:
                        inputs4[i] = np.append(inputs4[i], normalize(y[j] + dy3[j], norms[0, j], norms[1, j]), 1)
        # if b.shape[1] > 0:
        #     for k in range(len(y)):
        #         inputs4[k] = np.append(inputs4[k], othinputs[k])
        for ii in range(len(y0)):
            if np.amax(used_inputs[ii]) > 1:
                inputs4[ii] = reorder(used_inputs[ii], inputs4[ii])
        dy4 = prediction(inputs4) * h
        for p in range(len(y)):
            if (y[p] + dy3[p]) >= norms[1, p] and dy4[p] > 0:
                dy4[p] = 0
            if (y[p] + dy3[p]) <= norms[0, p] and dy4[p] < 0:
                dy4[p] = 0

        rdy1 = dy1.reshape(5, 1)
        rdy2 = dy2.reshape(5, 1)
        rdy3 = dy3.reshape(5, 1)
        rdy4 = dy4.reshape(5, 1)
        y += (rdy1 + 2 * rdy2 + 2 * rdy3 + rdy4) / 6
        Y = np.append(Y, y, 1)
        ind += 1

    return T, Y


