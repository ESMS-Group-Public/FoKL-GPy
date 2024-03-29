from src.FoKL import FoKLRoutines
import numpy as np
import tifffile
import os


def main():

    # Load dataset:

    try:
        inputs = np.load(os.path.join('dataset', 'inputs.npy'))
        data = np.load(os.path.join('dataset', 'data.npy'))

    except Exception as exception:
        vid = tifffile.imread(os.path.join('dataset', 'vid.tif'))

        n = []
        x = []
        for i in range(3):
            n.append(vid.shape[i])
            x.append(np.linspace(0, n[i] - 1, n[i]))
        uv = n[1] * n[2]
        tuv = n[0] * uv

        datatype = np.float16
        inputs = np.zeros([tuv, 3], dtype=datatype)
        data = np.zeros([tuv, 1], dtype=datatype)
        lo = 0
        for t in range(n[0]):
            hi = (t + 1) * uv
            data[lo:hi] = vid[t].reshape(-1, 1)  # [t, u, v] for t where [u, v] = [[0, 0], [0, 1], ..., [n[1], n[2]]
            inputs[lo:hi, :] = np.array([np.full(uv, t), np.repeat(x[1], n[2]), np.tile(x[2], n[1])]).T
            lo = hi

        np.save(os.path.join('dataset', 'inputs.npy'), inputs)
        np.save(os.path.join('dataset', 'data.npy'), data)

    # Clean dataset in FoKL model:

    try:
        model = FoKLRoutines.load(os.path.join('models', 'cleaned_16bit.fokl'))

    except Exception as exception:
        model = FoKLRoutines.FoKL()  # if from dataset, b=0 and btau=inf causes betas=nan
        model.clean(inputs, data, bit=16)
        model.save(os.path.join('models', 'cleaned_16bit.fokl'))

    # Fit:

    for train in [1e-6, 1e-4]:
        model.trainlog = model.generate_trainlog(train)
        model.fit()
        try:
            # model.save(f"fitted_{train}.fokl")
            np.save(os.path.join('models', f"betas_{train}.npy", model.betas))
            np.save(os.path.join('models', f"mtx_{train}.npy", model.mtx))
            np.save(os.path.join('models', f"evs_{train}.npy", model.evs))
        except Exception as exception:
            breakpoint()
        model.clear(keep=['inputs', 'data', 'normalize'])

    breakpoint()


if __name__ == '__main__':
    main()

    breakpoint()

