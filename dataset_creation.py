"""
Create a custom PyTorch Dataset for the MindBigDataImagenet data - a neurosignal dataset 
consisting of EEG signal measurements taken from subjects while they were presented with image stimuli from Imagenet.

 
Author: Hamza Javed
History: [Jun 2023]
"""

# Import relevant libraries
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Filepath locations of dataset in use
DATASET_ROOT = Path('/Users/hamza/Documents/Programming/pro_tutorials/Python/ML/Datasets/')
DATA_PATH = DATASET_ROOT.joinpath('MindBigData-Imagenet-IN/MindBigData-Imagenet/')

files = DATA_PATH.rglob('*.csv')  # create a generator containing all datafiles


# Build a custom Dataset
class MindBigData(Dataset):

    # These class variables used because each EEG recording a different length  #TODO: address this more rigorously
    start_sample = 0
    end_sample = 352  # corresponds to 2.75 seconds at 128 Hz sampling frequency

    def __init__(self, data_filepath, human_readable=True, transform=None):
        # Load data - but in the case of a large number of files just the file locations
        self.data_filepath = data_filepath
        self.human_readable = human_readable  # boolean to determine whether to use human-readable labels or synet codes
        self.eeg_channels = ['AF3', 'AF4', 'T7', 'T8', 'Pz']  # hardcoded assumes dataset is curated and consistent
        self.data_files_l = list(data_filepath.rglob('*.csv'))
        # self.data_labels_l = list(map(lambda x: self.parseLabel(x.stem, self.human_readable), self.data_files_l))
        self.n_samples = len(self.data_files_l)
        self.fs = 128  # sampling frequency in Hz
        self.label_code_dict = self.classLabels()  # links label codes to human readable labels
        self.label_distribution = self.labelDistribution()  # labels and corresponding counts present in the dataset
        self.transform = transform

    def __getitem__(self, item: int):
        # Access data, e.g. data[0]
        data = np.genfromtxt(self.data_files_l[item], delimiter=',')
        data = data[:, 1:]  # ignore index column
        label_code = self.parseLabel(Path(self.data_files_l[item]).stem, self.human_readable)
        if self.transform:
            return self.transform(data[:, MindBigData.start_sample:MindBigData.end_sample]), label_code
        else:
            return torch.tensor(data[:, MindBigData.start_sample:MindBigData.end_sample], dtype=torch.float32), label_code

    def __len__(self):
        # Return number of samples
        return self.n_samples

    def __repr__(self):
        # Dataset descriptor
        return f"\nMindBigData_Imagenet_Insight" \
               f"\n----------------------------" \
               f"\nConsisting of {self.n_samples} samples" \
               f"\n5-channel 3sec EEG data, in response to different Imagenet stimuli" \
               f'\nElectrodes in the 10/20 positions ({", ".join(chan for chan in self.eeg_channels)})'

    def parseLabel(self, filename: str, human_readable=False):
        # Return the label from the filename string
        label = filename.split('_')[3]  # hardcoded according to dataset file naming protocol
        if human_readable:
            label = self.classLabels()[label]
        return label

    def labelDistribution(self):  # TODO: should this be a private function (i.e. only meant to be used in class)?
        # Return counts of each type of class in the dataset as a dictionary
        label_dict = {}
        for file in self.data_files_l:
            sample_label = self.parseLabel(file.stem, self.human_readable)
            label_dict[sample_label] = label_dict.get(sample_label, 0) + 1
        return label_dict

    def classLabels(self, word_report_filepath=None):
        # Use human-readable class labels (not the alphanumeric code used from the synsnet of ILSVRC 2013)
        if not word_report_filepath:
            word_report_filepath = self.data_filepath.parent.joinpath('WordReport-v1.04.txt')
            assert word_report_filepath.is_file()

        df = pd.read_csv(word_report_filepath, delimiter='\t', names=['Descriptor', 'Count', 'Label Code'])
        code_label_dict = dict(zip(df['Label Code'], df['Descriptor'].str.split(',').str[0]))
        return code_label_dict

    def plotSignal(self, idx):
        # For visualisation purposes
        multichannel_data, sample_label_code = self[idx]  # load the i-th data and corresponding label
        plt.figure(figsize=(8, 5))
        plt.title(f"Example EEG data, sample {idx} from dataset - {sample_label_code.upper()} class")
        plt.plot(np.linspace(0, multichannel_data.shape[1]/self.fs, multichannel_data.shape[1]), multichannel_data.T)
        plt.legend(self.eeg_channels)
        plt.xlabel('Time (s)')
        plt.ylabel('EEG Amplitude')
        plt.show()


# Main body
# ---
dataset = MindBigData(DATA_PATH)
print(dataset)  # details about the dataset
dataset.plotSignal(0)  # example visualisation of one of the signals

# Train-test split
# train_idx, test_idx, train_y, test_y = train_test_split(list(range(dataset.n_samples)), dataset.data_labels_l, test_size=0.2, shuffle=True, stratify=dataset.data_labels_l)
# train_dataset = Subset(dataset)
#
# batch_size = 16
# dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
# features, label = next(iter(dataloader))  # for exploratory purposes