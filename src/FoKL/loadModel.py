import pickle
import os

def load(filename, directory=None):
    """
    Load a FoKL class from a file.

    By default, 'directory' is the current working directory that contains the script calling this method. An absolute
    or relative directory may be defined if the model to load is located elsewhere.

    For simplicity, enter the returned output from 'self.save()' as the argument here, i.e., for 'filename'. Do this
    while leaving 'directory' blank since 'filename' can simply include the directory itself.
    """
    if filename[-5::] != ".fokl":
        filename = filename + ".fokl"

    if directory is not None:
        filepath = os.path.join(directory, filename)
    else:
        filepath = filename

    file = open(filepath, "rb")
    model = pickle.load(file)
    file.close()

    return model