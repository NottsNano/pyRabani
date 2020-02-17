import os
import platform
import shutil
from datetime import datetime
from itertools import product

import h5py
import numpy as np

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
        self.params = None

        self.sweep_cnt = 1

    def call_rabani_sweep(self, params, axis_steps, image_reps):
        assert 0 not in kT_range, "Setting any value to 0 will cause buffer overflows and corrupted runs!"

        def get_linspace_ranges(ranges, axis_res):
            if type(ranges) is list:
                linspace = np.linspace(ranges[0], ranges[1], axis_res)
            else:
                linspace = [ranges]
            return linspace

        kT_linspace = get_linspace_ranges(params["kT"], axis_steps)
        mu_linspace = get_linspace_ranges(params["mu"], axis_steps)
        MR_linspace = get_linspace_ranges(params["MR"], axis_steps)
        C_linspace = get_linspace_ranges(params["C"], axis_steps)
        e_nl_linspace = get_linspace_ranges(params["e_nl"], axis_steps)
        e_nn_linspace = get_linspace_ranges(params["e_nn"], axis_steps)

        current_time = self.start_datetime.strftime("%H:%M:%S")
        print(f"{current_time} - Beginning generation of {axis_res * axis_res * image_reps} rabanis")

        self.params = np.zeros(((len(kT_linspace) * len(mu_linspace) * len(MR_linspace) * len(C_linspace) * len(
            e_nl_linspace) * len(e_nn_linspace)), 6))

        for kT_mus_cnt, (kT_val, mu_val, MR_val, C_val, e_nl_val, e_nn_val) in enumerate(
                product(kT_linspace, mu_linspace, MR_linspace, C_linspace, e_nl_linspace, e_nn_linspace)):
            self.params[kT_mus_cnt, :] = [kT_val, mu_val, MR_val, C_val, e_nl_val, e_nn_val]

        for image_rep in range(image_reps):
            imgs, m_all = _run_rabani_sweep(self.params)
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
                    f"{self.root_dir}/{self.start_date}/{self.start_time}/rabani_kT={self.params[rep, 0]:.2f}_mu={self.params[rep, 1]:.2f}_nsteps={int(m_all[rep]):d}_rep={image_rep}.txt",
                    img, fmt="%01d")
        elif self.savetype is "hdf5":
            for rep, img in enumerate(imgs):
                master_file = h5py.File(
                    f"{self.root_dir}/{self.start_date}/{self.start_time}/rabanis--{platform.node()}--{self.start_date}--{self.start_time}--{self.sweep_cnt}.h5",
                    "a")
                master_file.create_dataset("image", data=img, dtype="i1")
                master_file.attrs["kT"] = self.params[rep, 0]
                master_file.attrs["mu"] = self.params[rep, 1]
                master_file.attrs["MR"] = self.params[rep, 2]
                master_file.attrs["C"] = self.params[rep, 3]
                master_file.attrs["e_nl"] = self.params[rep, 4]
                master_file.attrs["e_nn"] = self.params[rep, 5]

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
    root_dir = "Images"
    total_image_reps = 1
    axis_res = 20

    kT_range = [0.01, 0.35]
    mu_range = [2.35, 3.5]
    MR_range = 1
    C_range = 0.4
    e_nl_range = 1.5
    e_nn_range = 2

    param_dict = {"kT": kT_range,
                  "mu": mu_range,
                  "MR": MR_range,
                  "C": C_range,
                  "e_nl": e_nl_range,
                  "e_nn": e_nn_range}

    rabani_sweeper = RabaniSweeper(root_dir=root_dir, savetype="hdf5")
    rabani_sweeper.call_rabani_sweep(params=param_dict,
                                     axis_steps=axis_res, image_reps=total_image_reps)
