"""Module for data organization
"""
import abc
import logging
from zipfile import ZipFile
import os
import shutil
import subprocess
from tqdm import tqdm
import requests
import rawpy
import numpy as np
import imageio
import bs4

ROOT='./data'


class Dataset(abc.ABC):
    """Abstract class for datasets: download, organize and clean

    Args:
        abc ([type]): [description]
    """

    def __init__(self, folder_name, root=ROOT):
        self.full_path = os.path.join(root, folder_name)
        self.ground_truth_folder = os.path.join(self.full_path, "ground_truth")
        self.inputs_folder = os.path.join(self.full_path, "inputs")
        os.makedirs(self.inputs_folder, exist_ok=True)
        os.makedirs(self.ground_truth_folder, exist_ok=True)

    @abc.abstractmethod
    def organize(self):
        """method for distributing the data on disk"""

    def clean(self):
        """method for cleaning the data"""
        shutil.rmtree(self.full_path)


class TestDataset(Dataset):
    """Based on having the file from
    https://alex04072000.github.io/SingleHDR/
    in the datasets folder
    """

    def organize(self, zfile):
        with ZipFile(zfile, "r") as zip_obj:
            zip_obj.extractall(path=self.full_path)
        temp_path = os.path.join(self.full_path, os.path.basename(zfile).split(".")[0])
        for folder in tqdm(os.listdir(temp_path)):
            temp_path_folder = os.path.join(temp_path, folder)
            gt_file = os.path.join(temp_path_folder, "gt.hdr")
            input_file = os.path.join(temp_path_folder, "input.jpg")
            new_gt_file = os.path.join(
                self.ground_truth_folder, "_".join([folder, "gt.hdr"])
            )
            new_input_file = os.path.join(
                self.inputs_folder, "_".join([folder, "input.jpg"])
            )
            shutil.copyfile(gt_file, new_gt_file)
            shutil.copyfile(input_file, new_input_file)
        shutil.rmtree(temp_path)


class SingleHDRDataset(Dataset):
    """Based on having the file from https://alex04072000.github.io/SingleHDR/
    in the datasets folder.
    For the moment it only includes the HDR Real part. Waiting to confirm
    that we can use HDR Synth
    """

    def organize(self, zfile):
        with ZipFile(zfile, "r") as zip_obj:
            zip_obj.extractall(path=self.full_path)
        temp_path = os.path.join(self.full_path, os.path.basename(zfile).split(".")[0])
        temp_path_real_gt = os.path.join(temp_path, "HDR-Real", "HDR_gt")
        temp_path_real_inputs = os.path.join(temp_path, "HDR-Real", "LDR_in")
        for input_file, ground_file in tqdm(
            zip(os.listdir(temp_path_real_inputs), os.listdir(temp_path_real_gt)),
            total=len(os.listdir(temp_path_real_gt)),
        ):
            input_file_name = os.path.join(temp_path_real_inputs, input_file)
            input_new_name = os.path.join(self.inputs_folder, input_file)
            gt_file_name = os.path.join(temp_path_real_gt, ground_file)
            gt_new_name = os.path.join(self.ground_truth_folder, ground_file)
            shutil.copyfile(input_file_name, input_new_name)
            shutil.copyfile(gt_file_name, gt_new_name)
        shutil.rmtree(temp_path)


class FuntDataset(Dataset):
    """Class for Funt Dataset
        https://www2.cs.sfu.ca/~colour/data/funt_hdr/
    """
    URL = "http://www.cs.sfu.ca/~colour/data2/funt_hdr/HDR.zip"

    @staticmethod
    def download(folder="../"):
        """Download the dataset and place it in folder
            check first if it has already been downloaded
        """
        name = os.path.join(folder, FuntDataset.URL.split("/")[-1])
        if os.path.isfile(name):
            logging.warning("File already downloaded")
        else:
            download_file(FuntDataset.URL, name)

    def organize(self, zfile):
        with ZipFile(zfile, "r") as zip_obj:
            zip_obj.extractall(path=self.full_path)
        temp_path = os.path.join(self.full_path, "cs")
        temp_path_real_gt = os.path.join(temp_path, "chroma", "data",
                                "Nikon_D700", "HDR_MATLAB_3x3")
        for ground_file in os.listdir(temp_path_real_gt):
            if not "CC" in ground_file:
                gt_file_name = os.path.join(temp_path_real_gt, ground_file)
                gt_new_name = os.path.join(self.ground_truth_folder, ground_file)
                shutil.copyfile(gt_file_name, gt_new_name)
        shutil.rmtree(temp_path)


class WardDataset(Dataset):
    """Class for Funt Dataset
        https://www2.cs.sfu.ca/~colour/data/funt_hdr/
    """

    def __init__(self, folder_name, root=ROOT):
        super().__init__(folder_name, root=root)
        self.url = "http://www.anyhere.com/gward/hdrenc/pages/originals.html"
        self.root = "http://www.anyhere.com/gward/hdrenc/pages/"

    def _download(self):
        """Download the dataset and place it in ground truth
            check first if it has already been downloaded
        """
        html = requests.get(self.url).content
        soup = bs4.BeautifulSoup(html)
        url_images = [i.get("href") for i in soup.find_all("a") if ".hdr" in i.get("href")]
        for i in url_images:
            name = os.path.join(self.ground_truth_folder, os.path.basename(i))
            if os.path.isfile(name):
                logging.warning("File already downloaded")
            else:
                download_file(
                    os.path.join(self.root, i), name)

    def organize(self):
        self._download()


class PFSToolsDataset(WardDataset):
    def __init__(self, folder_name, root=ROOT):
        super().__init__(folder_name, root=root)
        self.url = "http://pfstools.sourceforge.net/hdr_gallery.html"
        self.root = "http://pfstools.sourceforge.net"


class HDRPlusDataset(Dataset):
    """HDR+ dataset
    https://console.cloud.google.com/storage/browser/hdrplusdata/20171106?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false
    """

    folder_1_name = "./results_20171023/"

    @staticmethod
    def download():
        """Download the dataset and place it in folder
            check first if it has already been downloaded
        """
        subprocess.run("""
        gsutil -m cp -r \
            "gs://hdrplusdata/20171106/results_20171023/" \
            .
        """, shell=True)

    def organize(self):
        folders = os.listdir(HDRPlusDataset.folder_1_name)
        for j, folder in tqdm(enumerate(folders),
                             total=len(folders)):
            if any(map(str.isdigit, folder)):
                gt_file_name = os.path.join(
                    HDRPlusDataset.folder_1_name, folder, "merged.dng"
                )
                with rawpy.imread(gt_file_name) as raw:
                    image = raw.postprocess().astype(np.float32)
                gt_new_name = os.path.join(self.ground_truth_folder, "merged_" + str(j) + ".hdr")
                imageio.imsave(gt_new_name, image, format="HDR-FI")
        shutil.rmtree(HDRPlusDataset.folder_1_name)


validation_list = [
    'merged_228', 'merged_1426', 'merged_2824', 'merged_2084', 'merged_2297', 'merged_490', 'merged_481', 'merged_1959',
    'merged_1475', 'merged_852', 'merged_2636', 'merged_3129', 'merged_556', 'merged_2229', 'merged_2249', 'merged_2514',
    'merged_3177', 'merged_1062', 'merged_2846', 'merged_2162', 'merged_2409', 'merged_2929', 'merged_2313', 'merged_1407',
    'merged_837', 'merged_1942', 'merged_527', 'merged_2231', 'merged_1008', 'merged_2893', 'merged_1837', 'merged_724',
    'merged_1281', 'merged_1700', 'merged_884', 'merged_239', 'merged_637', 'merged_3532', 'merged_1596', 'merged_129',
    'merged_1501', 'merged_249', 'merged_344', 'merged_2315', 'merged_820', 'merged_1743', 'merged_2058', 'merged_2151',
    'merged_2515', 'merged_784', 'merged_2765', 'merged_2803', 'merged_978', 'merged_3169', 'merged_1513', 'merged_918',
    'merged_161', 'merged_1835', 'merged_1100', 'merged_2230', 'merged_1803', 'merged_261', 'merged_3448', 'merged_203',
    'merged_2730', 'merged_1394', 'merged_2750', 'merged_1352', 'merged_2157', 'merged_3404', 'merged_520', 'merged_599',
    'merged_1055', 'merged_3627', 'merged_278', 'merged_2154', 'merged_7', 'merged_1007', 'merged_1957', 'merged_2685',
    'merged_3271', 'merged_1676', 'merged_1474', 'merged_51', 'merged_3330', 'merged_3', 'merged_130', 'merged_2380',
    'merged_1965', 'merged_1463', 'merged_684', 'merged_452', 'merged_3237', 'merged_1497', 'merged_960', 'merged_3176',
    'merged_808', 'merged_1133', 'merged_750', 'merged_3329', 'merged_671', 'merged_117', 'merged_1437', 'merged_478',
    'merged_3574', 'merged_1773', 'merged_2174', 'merged_2562', 'merged_2259', 'merged_1000', 'merged_2855', 'merged_2780',
    'merged_847', 'merged_3620', 'merged_3222', 'merged_1490', 'merged_3599', 'merged_1883', 'merged_1301', 'merged_2178',
    'merged_3003', 'merged_2937', 'merged_3048', 'merged_3131', 'merged_2022', 'merged_3096', 'merged_1092', 'merged_611',
    'merged_3634', 'merged_1794', 'merged_1262', 'merged_1758', 'merged_928', 'merged_2172', 'merged_916', 'merged_3234',
    'merged_2917', 'merged_2526', 'merged_3505', 'merged_2176', 'merged_687', 'merged_1845', 'merged_3528', 'merged_589',
    'merged_2293', 'merged_3452', 'merged_1967', 'merged_1917', 'merged_1923', 'merged_1122', 'merged_2692', 'merged_323',
    'merged_793', 'merged_2762', 'merged_3461', 'merged_2401', 'merged_1576', 'merged_2720', 'merged_3032', 'merged_1624',
    'merged_877', 'merged_1814', 'merged_283', 'merged_2857', 'merged_2933', 'merged_1164', 'merged_1662', 'merged_2652',
    'merged_1626', 'merged_1335', 'merged_1863', 'merged_165', 'merged_1733', 'merged_917', 'merged_99', 'merged_1847',
    'merged_3145', 'merged_135', 'merged_674', 'merged_3296', 'merged_2975', 'merged_2432', 'merged_187', 'merged_3356',
    'merged_1072', 'merged_1531', 'merged_991', 'merged_3421'
    ]


class ValidationDataset(Dataset):

    def __init__(self, folder_name, source_folder, root=ROOT, size=0.05):
        super().__init__(folder_name, root=root)
        folder_path = os.path.join(root, source_folder, "ground_truth")
        self.files = os.listdir(folder_path)
        self.validation_list = validation_list
        self.folder_path = folder_path

    def organize(self):
        for elem in self.files:
            for val in self.validation_list:
                if val + "." in elem:
                    shutil.move(os.path.join(self.folder_path, elem),
                            os.path.join(self.ground_truth_folder, elem))


class DatasetCollection(Dataset):
    """Collection of datasets of Dataset type
    Args:
        class_list (list): list of classes of type Dataset
    """
    def __init__(self, folder_name, class_list, root=ROOT):
        super().__init__(folder_name, root=root)
        self.datasets = class_list

    def organize(self):
        total = 0
        for ds_class in self.datasets:
            for elem in tqdm(os.listdir(ds_class.ground_truth_folder),
                                total=len(os.listdir(ds_class.ground_truth_folder))):
                shutil.move(os.path.join(ds_class.ground_truth_folder, elem),
                            os.path.join(self.ground_truth_folder, str(total) + elem ))
                total += 1


def download_file(url: str, fname: str):
    """Download files with progress bar from 
    https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
    Args:
        url (str): url to download the file from
        fname (str): name to save the file wirh
    """
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


