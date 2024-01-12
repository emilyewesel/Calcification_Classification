from torch.utils.data import Dataset
import pandas as pd
import torch
import os
import nibabel as nib
import torchio as tio
import pytorch_lightning as pl
import numpy as np
from sklearn.model_selection import train_test_split

from settings import CSV_FILE, IMAGE_PATH, IMAGE_SIZE, VAL_SIZE, TEST_SIZE, FEATURES, TARGET, BATCH_SIZE, transformation, target_transformations
from torch.utils.data import DataLoader
from scipy.interpolate import interpn
from sklearn.preprocessing import MinMaxScaler

def resize(mat, new_size, interp_mode='linear'):
    """
    resize: resamples a "matrix" of spatial samples to a desired "resolution" or spatial sampling frequency via interpolation
    Args:
        mat:                matrix to be "resized" i.e. resampled
        new_size:         desired output resolution
        interp_mode:        interpolation method
    Returns:
        res_mat:            "resized" matrix
    """
    
    mat = mat.squeeze()
    mat_shape = mat.shape

    axis = []
    for dim in range(len(mat.shape)):
        dim_size = mat.shape[dim]
        axis.append(np.linspace(0, 1, dim_size))

    new_axis = []
    for dim in range(len(new_size)):
        dim_size = new_size[dim]
        new_axis.append(np.linspace(0, 1, dim_size))

    points = tuple(p for p in axis)
    xi = np.meshgrid(*new_axis)
    xi = np.array([x.flatten() for x in xi]).T
    new_points = xi
    mat_rs = np.squeeze(interpn(points, mat, new_points, method=interp_mode))
    # TODO: fix this hack.
    if dim + 1 == 3:
        mat_rs = mat_rs.reshape([new_size[1], new_size[0], new_size[2]])
        mat_rs = np.transpose(mat_rs, (1, 0, 2))
    else:
        mat_rs = mat_rs.reshape(new_size, order='F')
    # update command line status
    assert mat_rs.shape == tuple(new_size), "Resized matrix does not match requested size."
    return mat_rs
    
    
class NCANDADataset(Dataset):
    def __init__(self, image_dir, input_tabular, transform, target_transform):
        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform
        self.input_tab = input_tabular
        self.y = self.input_tab[TARGET]
        self.X = self.input_tab[FEATURES]

    def __len__(self):
        return len(self.input_tab)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        # print(f'{self.csv_df_split.iloc[idx, 0]}\n')
        image_name = os.path.join(self.image_dir, self.input_tab.iloc[idx, 0])

        subject_id = self.input_tab.iloc[idx, 0]

        image_path = image_name + '.nii.gz'

        image = nib.load(image_path)
        image = image.get_fdata()

        # change to numpy
        image = np.array(image, dtype=np.float32)

        # scale images between [0,1]
        image = image / image.max()

        image = resize(image, (IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE))

        if (self.transform and np.random.choice([0, 1]) == 0 ):
            transform = tio.RandomAffine(
            scales=(0.9, 1.2),
            degrees=10,
            )
            image = torch.tensor(image)

            # Add a singleton channel dimension to convert it to 4D
            image = torch.unsqueeze(image, 0)

            # Convert the Torch tensor to a TorchIO ScalarImage
            tio_image = tio.ScalarImage(tensor=image)

            # Apply the transformation to the image
            transformed_image = transform(tio_image)

            # Access the transformed image as a Torch tensor
            image = transformed_image.data.squeeze(0)

            # Convert the Torch tensor back to a NumPy array
            image = image.numpy()
            
            # temp = img[0]
            # image_data_new = temp.reshape(1, 64, 64, 64)
            # transformed = transform(image_data_new)
            # img = transformed

        
        label = self.y.values[idx]
        tab = self.X.values[idx]

        if self.target_transform:
            label = self.target_transform(label)

        return image, tab, label, subject_id


class NCANDADataModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()

    def get_stratified_split(self, csv_file):
        csv_df = pd.read_csv(csv_file)
        all_labels = csv_df[TARGET]
        subjects = csv_df['subject']
        all_labels = np.array(all_labels)
        train_subj, test_subj, y_train, y_test = train_test_split(subjects, all_labels, stratify=all_labels)

        return train_subj, test_subj, y_train, y_test
        
    
    def normalize_tabular_data(self, train_index, test_index, csv_file):

        scaler = MinMaxScaler(feature_range=(-1, 1))

        csv_df = pd.read_csv(csv_file)

        train_subjects = csv_df.loc[csv_df['subject'].isin(train_index)]
        test_subjects = csv_df.loc[csv_df['subject'].isin(test_index)]

        # we don't want to normalize based on subject, negative valence and depressive symptoms
        X_train = train_subjects[FEATURES]
        X_train = scaler.fit_transform(X_train)

        X_train = pd.DataFrame(data=X_train, columns=FEATURES)
        X_train = X_train.set_index(train_subjects.index)
        X_train.insert(0, 'subject', train_subjects.loc[:, 'subject'], True)
        X_train.insert(1, TARGET, train_subjects.loc[:, TARGET], True)

        X_test = test_subjects[FEATURES]
        X_test = scaler.transform(X_test)

        X_test = pd.DataFrame(data=X_test, columns=FEATURES)
        X_test = X_test.set_index(test_subjects.index)
        X_test.insert(0, 'subject', test_subjects.loc[:, 'subject'], True)
        X_test.insert(1, TARGET, test_subjects.loc[:, TARGET], True)

        return X_train, X_test, scaler
    
    def calculate_class_weight(self, X_train):

        y_train = X_train.loc[:, TARGET]
        number_neg_samples = np.sum(y_train.values == False)
        num_pos_samples = np.sum(y_train.values == True)
        mfb = number_neg_samples / num_pos_samples

        self.class_weights = mfb

        return mfb

    def prepare_data(self):

        train_subj, test_subj, y_train, y_test = self.get_stratified_split(CSV_FILE)
        
        X_train, X_test, self.scaler = self.normalize_tabular_data(train_subj, test_subj, CSV_FILE)
        
        self.class_weight = self.calculate_class_weight(X_train)

        self.train = NCANDADataset(image_dir=IMAGE_PATH,
                                   input_tabular=X_train, transform=transformation,
                                   target_transform=target_transformations)

        print(f'Train dataset length: {self.train.__len__()}')

        self.validation = NCANDADataset(image_dir=IMAGE_PATH,
                                        input_tabular=X_test, transform=transformation,
                                        target_transform=target_transformations)
                                        
        print(f'Validation dataset length: {self.validation.__len__()}')
        self.test = self.validation

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last = False)

    def val_dataloader(self):
        return DataLoader(self.validation, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, drop_last = True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, drop_last = True)

class Multimodal_Dataset(Dataset):

    def __init__(self, csv_dir, image_base_dir, target, features, categorical=None, transform=None, target_transform=None):
        """

        csv_dir: The directiry for the .csv file (tabular data) including the labels

        image_base_dir:The directory of the folders containing the images

        transform:The trasformations for the input images

        Target_transform:The trasformations for the target(label)

        """
        # TABULAR DATA
        # read .csv to load the data
        self.multimodal = pd.read_csv(csv_dir)

        # save categorical features in a variable
        self.categorical = categorical

        # set the target variable name
        self.target = target

        # set the dummy variables
        self.features = features

        # keep relevant features
        self.tabular = self.multimodal[self.features]

        # # one-hot encoding of the categorical variables
        # self.tabular_processed = pd.get_dummies(
        #     self.tabular, prefix=self.categorical)

        # Save target and predictors
        self.X = self.tabular.drop(self.target, axis=1)
        self.y = self.tabular[self.target]

        # IMAGE DATA
        self.imge_base_dir = image_base_dir

        self.transform = transform

        self.target_transform = target_transform

    def __len__(self):

        return len(self.tabular)

    def __getitem__(self, idx):

        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        label = self.y[idx]

        tab = self.X.iloc[idx].values

        # get image name in the given index
        img_folder_name = self.multimodal['Image_id'][idx]

        img_path = os.path.join(
            self.imge_base_dir, img_folder_name, 'image.nii.gz')

        image = nib.load(img_path)
        image = image.get_fdata()

        # change to numpy
        image = np.array(image, dtype=np.float32)

        # scale images between [0,1]
        image = image / image.max()

        return image, tab, label


class MultimodalDataModule(pl.LightningDataModule):

    def __init__(self):

        super().__init__()

    def prepare_data(self):

        self.train = Multimodal_Dataset(csv_dir=CSV_FILE + r'\train.csv', image_base_dir=IMAGE_PATH,
                                        target=TARGET, features=FEATURES,
                                        transform=transformation, target_transform=target_transformations)

        self.valid = Multimodal_Dataset(csv_dir=CSV_FILE + r'\val.csv', image_base_dir=IMAGE_PATH,
                                        target=TARGET, features=FEATURES,
                                        transform=transformation, target_transform=target_transformations)

        self.test = Multimodal_Dataset(csv_dir=CSV_FILE + r'\test.csv', image_base_dir=IMAGE_PATH,
                                       target=TARGET, features=FEATURES,
                                       transform=transformation, target_transform=target_transformations)

        # self.train, self.valid = torch.utils.data.random_split(
        #     self.train, [TRAIN_SIZE, VAL_SIZE + TEST_SIZE])

        # self.valid, self.test = torch.utils.data.random_split(
        #     self.valid, [VAL_SIZE, TEST_SIZE])

    def train_dataloader(self):

        return DataLoader(self.train, batch_size=2, shuffle=True)

    def val_dataloader(self):

        return DataLoader(self.valid, batch_size=1, shuffle=False)

    def test_dataloader(self):

        return DataLoader(self.test, batch_size=1, shuffle=False)
