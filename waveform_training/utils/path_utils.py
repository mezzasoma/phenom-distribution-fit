import os

def create_and_set_outdir(subfolder_name):
    """
    Create an output directory under the current working directory.

    Parameters:
    subfolder_name (str): Name of the subdirectory to create inside 'outdir'.
    """
    outdir = os.path.join(os.getcwd(), 'outdir', subfolder_name)
    os.makedirs(outdir, exist_ok=True)
    return outdir