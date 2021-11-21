import tarfile, zipfile, tempfile, shutil, gzip, os
from torch._six import PY37

def extract_archive(from_path, to_path=None, remove_finished=False):
    if to_path is None:
        base_path = '/'.join(from_path.split('/')[:-1])
        os.mkdir('/'.join([base_path,from_path.split('/')[-1].split('.')[0]]))
        to_path = '/'.join([base_path,from_path.split('/')[-1].split('.')[0]])
        #to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        with tarfile.open(from_path, 'r') as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path):
        with tarfile.open(from_path, 'r:gz') as tar:
            tar.extractall(path=to_path)
    elif _is_tarxz(from_path) and PY37:
        # .tar.xz archive only supported in Python 3.x
        with tarfile.open(from_path, 'r:xz') as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(to_path, os.path.splitext(os.path.basename(from_path))[0])
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, 'r') as z:
            z.extractall(to_path)
    else:
        raise ValueError(f"Extraction of {from_path} not supported")

    if remove_finished:
        os.remove(from_path)

    return to_path

def _is_targz(filename):
    return filename.endswith(".tar.gz")


def _is_tarxz(filename):
    return filename.endswith(".tar.xz")


def _is_gzip(filename):
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_tar(filename):
    return filename.endswith(".tar")


def _is_zip(filename):
    return filename.endswith(".zip")
