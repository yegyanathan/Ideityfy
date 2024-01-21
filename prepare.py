import os
import cv2
import string
import random
import numpy as np
import splitfolders

from PIL import Image
from tqdm.auto import tqdm
from cleanvision.imagelab import Imagelab
from os import path
import argparse
from huggingface_hub import HfApi

DATASET_DIRECTORY = path.join(path.dirname(__file__), "datasets")


class DataPipeline:
    """ Data processing pipeline for PyTorch image classification dataset."""

    def __init__(self, ds_name: str, do_rename: bool = False, do_split: bool = False, bypass: bool = False, do_print: bool = False, 
                 output: str = "deities", seed: int = 42, push_to_hf: bool = True, repo_id: str = "Yegiiii/deities"):
        
        self.ds_name = ds_name
        self.ds_path = os.path.join(DATASET_DIRECTORY, ds_name)
        assert os.path.exists(self.ds_path), f"Dataset {ds_name} not found in {DATASET_DIRECTORY}."
        # Data Pipeline
        self.filepaths = [] 
        for  dirpath, _, filenames in os.walk(self.ds_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                self.filepaths.append(filepath)

        self.do_rename = do_rename
        self.do_split = do_split
        self.bypass = bypass
        self.do_print = do_print
        self.output = output
        self.seed = seed
        self.push_to_hf = push_to_hf
        self.repo_id = repo_id

    
    def __call__(self) -> None:
        """ Pipeline """ 

        if self.do_rename:
            self.rename_files() 

        if self.bypass:
            self.check_file_integrity()
            self.convert_file_to_rgb() 
            self.clean_image_files() 

        if self.do_print:
            self.print_stats()

        if self.do_split:
            self.generate_split()

        if self.push_to_hf:
            api = HfApi()
            api.upload_folder(repo_id=self.repo_id, 
                              folder_path=self.ds_path, repo_type="dataset")


    def generate_split(self, ratio=(0.8, 0.2)) -> None:
        """ Splits the dataset based on the split ratio. """
        splitfolders.ratio(self.ds_path, ratio=ratio, group_prefix=None, output=self.output) 


    def check_file_integrity(self) -> None:
        """ Checks the integrity of image files and remove the invalid ones. """
        invalid_files = []
        for filepath in tqdm(self.filepaths):
            if not filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
                tqdm.write(f'file {os.path.basename(filepath)}  has an invalid extension.')
                invalid_files.append(filepath)  
            else:
                try:
                    img = cv2.imread(filepath)
                    _ = img.shape
                except:
                    tqdm.write(f'file {os.path.basename(filepath)} is not a valid image file ')
                    invalid_files.append(filepath)

        for filepath in invalid_files:
            os.remove(filepath)
            self.filepaths.remove(filepath)


    def rename_files(self) -> None:
        """ Renames the files for uniformity. """
        for filepath in tqdm(self.filepaths):
            dir_path = os.path.dirname(filepath)
            new_name = "".join(
                random.choice(
                    string.ascii_letters + string.digits) for i in range(8))
            new_name = new_name + os.path.splitext(filepath)[1]
            new_filepath = os.path.join(dir_path, new_name)
            os.rename(filepath, new_filepath)
            self.filepaths.remove(filepath)
            self.filepaths.append(new_filepath)


    def convert_file_to_rgb(self) -> None:
        """ RGBises the images """
        for filepath in tqdm(self.filepaths):
            img = Image.open(filepath)
            try:
                imgarr = np.asarray(img)
                imgarr[:, :, 2].astype("int")
            except:
                img = img.convert('RGB')
                img.save(filepath)


    def clean_image_files(self) -> None:
        """ Finds issues with the dataset and removes the bad ones. """
        bad_files = []
        imagelab = Imagelab(filepaths=self.filepaths)
        imagelab.find_issues()
        imagelab.report()

        issues = imagelab.issue_summary["issue_type"].tolist()
        for issue in issues:
            if issue in ("near_duplicates", "exact_duplicates"):
                sets = imagelab.info[issue]['sets']
                for set in sets:
                    bad_files.append(set[0])
            else:
                issue_key = f"is_{issue}_issue"
                score_key = f"{issue}_score"
                bad_files.extend(imagelab.issues[imagelab.issues[issue_key] == True].sort_values(by=[score_key]).index.tolist())

        print("Bad image files count: {} ".format(len(bad_files)))
        print("Bad image files: {} ".format(bad_files))

        for filepath in bad_files:
            os.remove(filepath)
            self.filepaths.remove(filepath)


    def print_stats(self) -> None:
        """ Prints dataset statistics. """
        deities = os.listdir(self.ds_path)
        tot_count = 0
        print("\nTotal number of classes: {}\n".format(len(deities)))
        print("Name of the class: \t\t Total number of samples:")
        for deity in deities:
            count = len(os.listdir(os.path.join(self.ds_path, deity)))
            tot_count += count
            print("{}\t\t\t\t\t{}".format(deity, count))
        print("\nTotal number of samples in the dataset: {}\n".format(tot_count))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--ds-name", default="deities", type=str, required=True, help='Dataset name (default: "deities")')
    parser.add_argument('--do-rename', action='store_true', default=False, help='Flag for renaming files (default: True)')
    parser.add_argument('--do-split', action='store_true', default=False, help='Flag for splitting the dataset into Train and Val (default: True)')
    parser.add_argument('--do-print', action='store_true', default=False, help='Flag for printing statistics (default: True)')
    parser.add_argument('--output', type=str, default="deities", help='Output dir name (default: "deities")')
    parser.add_argument('--seed', type=int, default=42, help='Seed value (default: 42)')
    parser.add_argument('--push-to-hf', action='store_true', default=False, help='Flag for pushing dataset to Huggingface (default: True)')

    args = parser.parse_args()

    pipeline = DataPipeline(ds_name=args.ds_name, do_rename=args.do_rename, do_split=args.do_split, 
                            do_print=args.do_print, output=args.output, seed=args.seed)
    
    pipeline()

