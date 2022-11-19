from datasets import load_dataset
import subprocess
import shutil
import sys
import os

def onerror(func, path, exc_info):
    """
    Error handler for ``shutil.rmtree``.

    If the error is due to an access error (read only file)
    it attempts to add write permission and then retries.

    If the error is for another reason it re-raises the error.
    
    Usage : ``shutil.rmtree(path, onerror=onerror)``
    """
    import stat
    # Is the error an access error?
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise

# call git clone https://github.com/IndoNLP/nusa-crowd.git
subprocess.call("git clone https://github.com/IndoNLP/nusa-crowd.git", shell=True)

sys.path.append("nusa-crowd")
path = "nusa-crowd/nusacrowd/nusa_datasets/id_stance"
dataset = load_dataset(path)

df = dataset['train'].to_pandas()

output_path = "./raw/data.csv"
df.to_csv(output_path, index=False)

# delete repository
shutil.rmtree("nusa-crowd", onerror=onerror)