import os
import numpy as np

from torch.utils.data import Dataset


class MMFiDataset(Dataset):

    def __init__(
        self,
        data_list,
        perturbation=None,
        return_complex=False
    ):
        """
        Args:
            data_list:
                CSI 파일 경로 리스트

            perturbation:
                noise/interference module

            return_complex:
                complex 유지 여부
        """

        self.data_list = data_list

        self.perturbation = perturbation

        self.return_complex = return_complex

    def __len__(self):

        return len(self.data_list)

    def load_csi(self, path):

        """
        CSI 로딩 함수

        예시:
            .npy
            .npz
            .mat
        형태에 맞게 수정
        """

        csi = np.load(path)

        return csi

    def preprocess(self, csi):

        """
        필요 시 전처리
        """

        # complex → amplitude only 예시
        if not self.return_complex:

            csi = np.abs(csi)

        return csi

    def __getitem__(self, idx):

        # ====================================
        # path
        # ====================================

        path = self.data_list[idx]

        # ====================================
        # load
        # ====================================

        csi = self.load_csi(path)

        # ====================================
        # perturbation
        # ====================================

        if self.perturbation is not None:

            csi = self.perturbation(csi)

        # ====================================
        # preprocess
        # ====================================

        csi = self.preprocess(csi)

        # ====================================
        # float32
        # ====================================

        csi = csi.astype(np.float32)

        return csi