import os
import platform
import shutil
import time
from datetime import datetime

import h5py
import numpy as np
from matplotlib import colors

from rabani import _run_rabani_sweep


class RabaniSweeper:
    def __init__(self, root_dir, savetype="hdf5", zip_when_done=False):
        self.system_name = platform.node()
        self.savetype = savetype
        self.root_dir = root_dir
        self.zip_when_done = zip_when_done

        self.start_datetime = datetime.now()
        self.start_date = self.start_datetime.strftime("%Y-%m-%d")
        self.start_time = self.start_datetime.strftime("%H-%M")
        self.end_datetime = None
        self.kT_mus = None

        self.sweep_cnt = 1

    def call_rabani_sweep(self, kT_range, mu_range, axis_steps, image_reps):
        assert 0 not in kT_range, "Setting any value to 0 will cause buffer overflows and corrupted runs!"

        current_time = self.start_datetime.strftime("%H:%M:%S")
        print(f"{current_time} - Beginning generation of {axis_res * axis_res * image_reps} rabanis")

        # Manually permute inputs (numba can't do itertools D:)
        self.kT_mus = np.zeros((axis_steps ** 2, 2))
        kT_mus_cnt = 0
        for kT_val in np.linspace(kT_range[0], kT_range[1], axis_res):
            for mu_val in np.linspace(mu_range[0], mu_range[1], axis_res):
                self.kT_mus[kT_mus_cnt, :] = [kT_val, mu_val]
                kT_mus_cnt += 1

        for image_rep in range(image_reps):
            imgs, m_all = _run_rabani_sweep(self.kT_mus)
            imgs = np.swapaxes(imgs, 0, 2)
            self.save_rabanis(imgs, m_all, image_rep)

            now = datetime.now().strftime("%H:%M:%S")
            print(
                f"{now} - Successfully completed block {image_rep + 1} of {image_reps} ({axis_res * axis_res} rabanis)")

        self.end_datetime = datetime.now()

        if self.zip_when_done:
            self.zip_rabanis()

    def make_storage_folder(self, dir):
        if not os.path.isdir(dir):
            os.makedirs(dir)

    def save_rabanis(self, imgs, m_all, image_rep):
        self.make_storage_folder(f"{self.root_dir}/{self.start_date}/{self.start_time}")
        if self.savetype is "txt":
            for rep, img in enumerate(imgs):
                np.savetxt(
                    f"{self.root_dir}/{self.start_date}/{self.start_time}/rabani_kT={self.kT_mus[rep, 0]:.2f}_mu={self.kT_mus[rep, 1]:.2f}_nsteps={int(m_all[rep]):d}_rep={image_rep}.txt",
                    img, fmt="%01d")
        elif self.savetype is "hdf5":
            for rep, img in enumerate(imgs):
                master_file = h5py.File(
                    f"{self.root_dir}/{self.start_date}/{self.start_time}/rabanis--{platform.node()}--{self.start_date}--{self.start_time}--{self.sweep_cnt}.h5",
                    "a")
                master_file.create_dataset("image", data=img, dtype="i1")
                master_file.attrs["kT"] = self.kT_mus[rep, 0]
                master_file.attrs["mu"] = self.kT_mus[rep, 1]
                master_file.attrs["num_mc_steps"] = m_all[rep]

                self.sweep_cnt += 1
        elif self.savetype is None:
            pass
        else:
            raise LookupError("Specified storage format not available")

    def zip_rabanis(self):
        shutil.make_archive(
            f"{self.root_dir}/{self.start_date}/{self.start_time}/rabanis--{platform.node()}--{self.start_date}--{self.start_time}--zipped.zip",
            'zip', f"{self.root_dir}/{self.start_date}/{self.start_time}")


if __name__ == '__main__':
    cmap = colors.ListedColormap(["black", "white", "orange"])
    boundaries = [0, 0.5, 1]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    start = time.time()
    root_dir = "Images"
    total_image_reps = 1
    axis_res = 50
    kT_range = [0.01, 0.5]
    mu_range = [2.15, 3.5]

    rabani_sweeper = RabaniSweeper(root_dir=root_dir, savetype="hdf5")
    rabani_sweeper.call_rabani_sweep(kT_range=kT_range, mu_range=mu_range,
                                     axis_steps=axis_res, image_reps=total_image_reps)
