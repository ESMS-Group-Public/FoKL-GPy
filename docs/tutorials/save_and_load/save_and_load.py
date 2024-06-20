"""
[TUTORIAL]: self.save(self, filename=None, directory=None)
            FoKL.Routines.load(filename, directory=None)

This is a tutorial for the 'save' method, a useful feature that allows saving a FoKL model. This is most beneficial
after training a large model, as 'fit' takes the longest to process.

In this tutorial, the following will be demonstrated:
    1) Train a new FoKL model.
    2) Save the model.
    3) Load a model.
"""
from FoKL import FoKLRoutines
import os
dir = os.path.abspath(os.path.dirname(__file__))  # directory of script
# # -----------------------------------------------------------------------
# # UNCOMMENT IF USING LOCAL FOKL PACKAGE:
# import sys
# sys.path.append(os.path.join(dir, '..', '..', '..'))  # package directory
# from src.FoKL import FoKLRoutines
# # -----------------------------------------------------------------------
import warnings


def main():
    print("\nThe following is an example of generating, saving, and loading a FoKL model.")

    # Define filename of model to save/load, and its directory:

    filename = "model.fokl"
    directory = os.path.join(dir, "folder_for_model")

    # (1) Train a new FoKL model:

    print("\nTraining model...\n")
    f = FoKLRoutines.FoKL()
    x = [1, 2, 3, 4, 5]
    y = [1, 4, 9, 16, 25]
    f.fit(x, y, clean=True)

    # (2) Save the model:

    print("\nSaving model...")
    filepath = f.save(filename, directory)

    # (3) Load the model:

    print('\nLoading model...')
    f_loaded = FoKLRoutines.load(filepath)  # = FoKLRoutines.load(filename, directory)

    # Compare saved and loaded models to confirm functionality:

    print("\nConfirming functionality...\n")
    if f.kernel == f_loaded.kernel and all(all(f.betas[draw, :] == f_loaded.betas[draw, :]) for draw in range(1000)) \
            and all(f.mtx == f_loaded.mtx):
        print("Success! The saving and loading functions are working properly.")
    else:
        warnings.warn("The saving and loading functions are NOT working properly.", category=UserWarning)


if __name__ == '__main__':
    main()
    print("\nEnd of save and load tutorial.")

