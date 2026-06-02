import os
import numpy as np

import torch

from scipy.io import loadmat
from torch.utils.data import Dataset


class MMFiDataset(Dataset):

    def __init__(
        self,
        env_dirs,
        target_actions,
        perturbation=None
    ):

        self.perturbation = perturbation

        self.samples = []

        print("📂 MMFi Dataset 스캔 시작")

        # ==========================================================
        # Environment Scan
        # ==========================================================

        for env_dir in env_dirs:

            if not os.path.exists(env_dir):

                print(f"⚠️ {env_dir} 없음")
                continue

            print(f"▶️ Scan: {env_dir}")

            for subj in sorted(os.listdir(env_dir)):

                subj_path = os.path.join(
                    env_dir,
                    subj
                )

                if not os.path.isdir(subj_path):
                    continue

                for action in sorted(os.listdir(subj_path)):

                    if action not in target_actions:
                        continue

                    action_path = os.path.join(
                        subj_path,
                        action
                    )

                    if not os.path.isdir(action_path):
                        continue

                    csi_folder = os.path.join(
                        action_path,
                        "wifi-csi"
                    )

                    gt_file = os.path.join(
                        action_path,
                        "ground_truth.npy"
                    )

                    if not os.path.exists(csi_folder):
                        continue

                    if not os.path.exists(gt_file):
                        continue

                    gt_data = np.load(gt_file)

                    csi_files = sorted(
                        os.listdir(csi_folder)
                    )

                    num_frames = min(
                        len(csi_files),
                        gt_data.shape[0]
                    )

                    # ==================================================
                    # Frame-level sample 저장
                    # ==================================================

                    for i in range(num_frames):

                        csi_path = os.path.join(
                            csi_folder,
                            csi_files[i]
                        )

                        pose = gt_data[i]

                        self.samples.append(
                            (
                                csi_path,
                                pose
                            )
                        )

        print(f"✅ 총 샘플 수: {len(self.samples)}")

    # ==============================================================
    # Length
    # ==============================================================

    def __len__(self):

        return len(self.samples)

    # ==============================================================
    # Get Item
    # ==============================================================

    def __getitem__(self, idx):

        csi_path, pose = self.samples[idx]

        # ==========================================================
        # CSI Load
        # ==========================================================

        mat_data = loadmat(csi_path)

        amp = np.nan_to_num(
            mat_data["CSIamp"]
        )

        phase = np.nan_to_num(
            mat_data["CSIphase"]
        )

        csi_complex = amp * np.exp(
            1j * phase
        )

        # ==========================================================
        # Perturbation
        # ==========================================================

        if self.perturbation is not None:
            if idx < 3:
                print(f"🔥 Noise Applied | idx={idx}")

            csi_complex = self.perturbation(
                csi_complex
            )

        # ==========================================================
        # Magnitude / Phase
        # ==========================================================

        csi_mag = np.abs(
            csi_complex
        ).reshape(-1)

        csi_ph = np.angle(
            csi_complex
        ).reshape(-1)

        csi = np.concatenate(
            [csi_mag, csi_ph],
            axis=0
        )

        # ==========================================================
        # Normalize
        # ==========================================================

        csi = np.nan_to_num(csi)

        csi_min = csi.min()
        csi_max = csi.max()

        csi = (
            csi - csi_min
        ) / (
            csi_max - csi_min + 1e-6
        )

        # ==========================================================
        # Tensor
        # ==========================================================

        csi_tensor = torch.tensor(
            csi,
            dtype=torch.float32
        )

        pose_tensor = torch.tensor(
            pose.reshape(-1),
            dtype=torch.float32
        )

        return csi_tensor, pose_tensor
