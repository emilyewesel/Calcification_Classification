from torch.utils.data import Dataset
import pandas as pd
import torch
import os
import nibabel as nib

import pytorch_lightning as pl
import numpy as np
from sklearn.model_selection import train_test_split

from settings import CSV_FILE, IMAGE_PATH, IMAGE_SIZE, VAL_SIZE, TEST_SIZE, FEATURES, TARGET, BATCH_SIZE, \
    transformation, target_transformations
from torch.utils.data import DataLoader
from scipy.interpolate import interpn
from sklearn.preprocessing import MinMaxScaler
import random


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


class NCANDADatasetTriplet(Dataset):
    def __init__(self, image_dir, input_tabular, transform, target_transform, age, grouped):
        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform
        self.input_tab = input_tabular
        self.y = self.input_tab[TARGET]
        self.X = self.input_tab[FEATURES]
        self.ages = age
        self.grouped = grouped
        self.triplet_by = 'age+disease'

    def __len__(self):
        return len(self.input_tab)

    def get_class_weight(self):
        return self.class_weight

    def get_sample(self, subject_id):
        # print(f'{self.csv_df_split.iloc[idx, 0]}\n')
        # image_name = os.path.join(self.image_dir, self.input_tab.iloc[idx, 0])

        image_name = os.path.join(self.image_dir, subject_id)

        image_path = image_name + '.nii.gz'

        image = nib.load(image_path)
        image = image.get_fdata()

        # change to numpy
        image = np.array(image, dtype=np.float32)

        # scale images between [0,1]
        image = image / image.max()
        # print(f'Old size: {np.shape(image)}')
        image = resize(image, (IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE)) #128, 128 or even 64
        # print(f'New size: {np.shape(image)}')
        subject_data = self.input_tab.loc[self.input_tab['subject'] == subject_id]
        label = subject_data['depressive_symptoms'].values[0]
        tab = subject_data[FEATURES].values[0,:]

        if self.target_transform:
            label = self.target_transform(label)

        return image, tab, label


    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        subject_id = self.input_tab.iloc[idx, 0]

        image, tab, label = self.get_sample(subject_id)
         # true as long as one is true
        neg_class = list(set(self.grouped.keys()) - set([int(label)]))[0]  # assuming only two classes

        # baseline triplet: sample a positive example from the same class
        if self.triplet_by is None:

            pos_ex_ID = random.sample(self.grouped[label], 1)[0]
            neg_ex_ID = random.sample(self.grouped[neg_class], 1)[0]
        elif self.triplet_by == 'age+disease':
            # print("triplet_by is age")
            subj_age = self.ages.iloc[idx]['visit_age']
            margin = 1
            # same class, not itself
            pos_group_all = self.ages[(self.ages['depressive_symptoms'] == label) & (self.ages['subject'] != subject_id)]
            pos_group = pos_group_all[(pos_group_all['visit_age'] > subj_age-margin) & (pos_group_all['visit_age'] < subj_age+margin)]
            while len(pos_group) < 1:
                # print('resampling positive example...')
                margin += 0.5
                pos_group = pos_group_all[(pos_group_all['visit_age'] > subj_age-margin) & (pos_group_all['visit_age'] < subj_age+margin)]
            pos_ex_ID = pos_group.sample()['subject'].iloc[0]
            margin = 1
            # different class
            neg_group_all = self.ages[(self.ages['depressive_symptoms'] == neg_class) & (self.ages['subject'] != subject_id)]
            neg_group = neg_group_all[(neg_group_all['visit_age'] > subj_age-margin) & (neg_group_all['visit_age'] < subj_age+margin)]
            while len(neg_group) < 1:
                # print('resampling negative example...')
                margin += 0.5
                neg_group = neg_group_all[(neg_group_all['visit_age'] > subj_age-margin) & (neg_group_all['visit_age'] < subj_age+margin)]
            neg_ex_ID = neg_group.sample()['subject'].iloc[0]

        pos_X, pos_tab, pos_label = self.get_sample(pos_ex_ID)

        neg_X, neg_tab, neg_label = self.get_sample(neg_ex_ID)

        return {'anchor': (image, tab, label, subject_id),
                'positive': (pos_X, pos_tab, pos_label, pos_ex_ID),
                'negative': (neg_X, neg_tab, neg_label, neg_ex_ID)}


class NCANDADataTripletModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()

    def get_stratified_split(self, csv_file):

        group_by_construct_train = {1: [], 0: []}
        group_by_construct_test = {1: [], 0: []}

        csv_df = pd.read_csv(csv_file)
        all_labels = csv_df[TARGET]
        subjects = csv_df['subject']
        all_labels = np.array(all_labels)
        train_subj, test_subj, y_train, y_test = train_test_split(subjects, all_labels, stratify=all_labels)

        train_subj_df = csv_df[csv_df['subject'].isin(list(train_subj))]
        age_train = train_subj_df[['subject', 'visit_age', 'depressive_symptoms']]

        test_subj_df = csv_df[csv_df['subject'].isin(list(test_subj))]
        age_test = test_subj_df[['subject', 'visit_age', 'depressive_symptoms']]

        for subject in train_subj:
            subj_visits = csv_df[csv_df['subject'] == subject]
            subj_label = subj_visits['depressive_symptoms']
            group_by_construct_train[subj_label.values[0]].append(subject)

        for subject in test_subj:
            subj_visits = csv_df[csv_df['subject'] == subject]
            subj_label = subj_visits['depressive_symptoms']
            group_by_construct_test[subj_label.values[0]].append(subject)

        return train_subj, test_subj, y_train, y_test, age_train, age_test, group_by_construct_train, group_by_construct_test

    def normalize_tabular_data(self, train_index, test_index, csv_file):

        scaler = MinMaxScaler(feature_range=(-1, 1))

        csv_df = pd.read_csv(csv_file)

        train_subjects = csv_df.loc[csv_df['subject'].isin(train_index)]
        test_subjects = csv_df.loc[csv_df['subject'].isin(test_index)]

        # we don't want to normalize based on subject and negative valence
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

        train_subj, test_subj, y_train, y_test, age_train, age_test, group_by_construct_train, group_by_construct_test = self.get_stratified_split(CSV_FILE)

        X_train, X_test, self.scaler = self.normalize_tabular_data(train_subj, test_subj, CSV_FILE)

        self.class_weight = self.calculate_class_weight(X_train)

        self.train = NCANDADatasetTriplet(image_dir=IMAGE_PATH,
                                   input_tabular=X_train, transform=transformation,
                                   target_transform=target_transformations,
                                   age=age_train,
                                   grouped=group_by_construct_train)

        print(f'Train dataset length: {self.train.__len__()}')

        self.validation = NCANDADatasetTriplet(image_dir=IMAGE_PATH,
                                        input_tabular=X_test, transform=transformation,
                                        target_transform=target_transformations,
                                        age=age_test,
                                        grouped=group_by_construct_test)

        print(f'Validation dataset length: {self.validation.__len__()}')

        self.test = self.validation

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.validation, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, drop_last=True)

