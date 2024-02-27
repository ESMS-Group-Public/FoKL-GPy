from src.FoKL import FoKLRoutines


def main():
    print("The following is an example of generating, saving, and loading a FoKL model.")
    print("----------------------------------------------------------------------------")

    # Define a filename and directory:
    filename = "model.fokl"
    directory = "folder_for_model"

    # Generate a model:
    print("Generating model...")
    f = FoKLRoutines.FoKL()
    x = [1, 2, 3, 4, 5]
    y = [1, 4, 9, 16, 25]
    f.fit(x, y, ConsoleOutput=False)

    # Save the model:
    print("Saving model...")
    filepath = f.save(filename, directory)

    # Load the model:
    print('Loading model...')
    f_loaded = FoKLRoutines.load(filepath)  # = FoKLRoutines.load(filename, directory)

    # Compare saved and loaded models:
    print("Confirming functionality...")
    if f.kernel == f_loaded.kernel and all(f.betas_avg == f_loaded.betas_avg) and all(f.mtx == f_loaded.mtx):
        print("Success! The saving and loading functions are working properly.")
    else:
        raise ValueError("The saving and loading functions are NOT working properly.")


if __name__ == '__main__':
    main()

