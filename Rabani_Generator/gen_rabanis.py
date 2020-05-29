import json
import os
import platform
import warnings
from datetime import datetime
from itertools import product

import h5py
import numpy as np
import paramiko
from scipy.stats import mode
from skimage import measure
from tqdm import tqdm

from Rabani_Generator.rabani import _run_rabani_sweep


class RabaniSweeper:
    def __init__(self, root_dir, generate_mode, sftp_when_done=False):
        self.system_name = platform.node()
        self.root_dir = root_dir

        self.generate_mode = generate_mode
        assert generate_mode in ["make_dataset", "visualise"]

        self.sftp_when_done = sftp_when_done
        self.ssh = None
        self.sftp = None

        self.start_datetime = datetime.now()
        self.start_date = self.start_datetime.strftime("%Y-%m-%d")
        self.start_time = self.start_datetime.strftime("%H-%M")
        self.end_datetime = None

        self.params = None
        self.sweep_cnt = 1

        self._dir_base = f"{self.root_dir}/{self.start_date}/{self.start_time}"
        self._file_base = f"{self._dir_base}/rabanis--{platform.node()}--{self.start_date}--{self.start_time}"
        self.make_storage_folder(self._dir_base)

    def setup_ssh(self):
        with open("details.json", 'r') as f:
            details = json.load(f)

        self.ssh = paramiko.SSHClient()
        self.ssh.load_system_host_keys()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(details["ip_addr"], username=details["user"], password=details["pass"])
        self.sftp = self.ssh.open_sftp()

    def call_rabani_sweep(self, params, axis_steps, image_reps):
        """Run an optimised set of rabani simulations, sweeping along desired axis/axes

        Parameters
        ----------
        params : dict[str | int or float] or dict[str | list[int or float, int or float] ]
            Parameters describing the values of kT, mu, MR, C, e_nl, e_nn and L of the simulations. Single values
            are fixed, while a list of [min max] will be swept through. L can only scale as 2^n.
        axis_steps : int or dict[str | int]
            Resolution of the sweep for each axis
        image_reps : int
            Number of repeats of each swept value
        """

        def get_linspace_ranges(param, param_key, axis_res):
            if type(param[param_key]) is list:
                if type(axis_res) is dict:
                    if param_key in axis_res.keys():
                        linspace = np.linspace(param[param_key][0], param[param_key][1], axis_res[param_key])
                    else:
                        linspace = [param[param_key]]
                else:
                    linspace = np.linspace(param[param_key][0], param[param_key][1], axis_res)
            else:
                linspace = [param[param_key]]

            return linspace

        def get_square_ranges(L_param, axis_res):
            if type(L_param) is list:
                if type(axis_res) is dict:
                    if "L" in axis_res.keys():
                        min_pwr = int(np.log(L_param[0]) / np.log(2))
                        max_pwr = int(np.log(L_param[1]) / np.log(2)) + 1
                        sqrspace = np.power(2, np.arange(min_pwr, max_pwr))
                    else:
                        sqrspace = [L_param]
                else:
                    min_pwr = int(np.log(L_param[0]) / np.log(2))
                    max_pwr = int(np.log(L_param[1]) / np.log(2))
                    sqrspace = np.power(2, np.arange(min_pwr, max_pwr))
            else:
                sqrspace = [L_param]

            return sqrspace

        kT_linspace = get_linspace_ranges(params, "kT", axis_steps)
        mu_linspace = get_linspace_ranges(params, "mu", axis_steps)
        MR_linspace = get_linspace_ranges(params, "MR", axis_steps)
        C_linspace = get_linspace_ranges(params, "C", axis_steps)
        e_nl_linspace = get_linspace_ranges(params, "e_nl", axis_steps)
        e_nn_linspace = get_linspace_ranges(params, "e_nn", axis_steps)
        L_all = np.array(list(map(int, get_linspace_ranges(params, "L", axis_steps))))

        tot_len = len(np.array(
            list(product(kT_linspace, mu_linspace, MR_linspace, C_linspace, e_nl_linspace, e_nn_linspace, L_all))))

        current_time = self.start_datetime.strftime("%H:%M:%S")
        print(f"{current_time} - Beginning generation of {tot_len * image_reps} rabanis")

        if self.generate_mode == "visualise":
            warnings.warn("Generation mode is currently set to visualisation!")
            assert image_reps == 1

        pbar = tqdm(total=tot_len * image_reps)
        for image_rep in range(image_reps):
            for L in L_all:
                params = np.array(
                    list(product(kT_linspace, mu_linspace, MR_linspace, C_linspace, e_nl_linspace, e_nn_linspace, [L])))
                assert 0. not in params, "Setting any value to 0 will cause buffer overflows and corrupted runs!"

                imgs, m_all = _run_rabani_sweep(params)
                self.save_rabanis(imgs, m_all, params)
                pbar.update(len(params))

        self.end_datetime = datetime.now()

    def make_storage_folder(self, dir):
        if not os.path.isdir(dir):
            os.makedirs(dir)

    def save_rabanis(self, imgs, m_all, params):
        for rep, img in enumerate(imgs):
            master_file = h5py.File(
                f"{self._file_base}--{self.sweep_cnt}.h5",
                "a")

            region, cat = self.calculate_stats(img, params[rep, 6])

            master_file.attrs["kT"] = params[rep, 0]
            master_file.attrs["mu"] = params[rep, 1]
            master_file.attrs["MR"] = params[rep, 2]
            master_file.attrs["C"] = params[rep, 3]
            master_file.attrs["e_nl"] = params[rep, 4]
            master_file.attrs["e_nn"] = params[rep, 5]
            master_file.attrs["L"] = params[rep, 6]
            master_file.attrs["category"] = cat

            sim_results = master_file.create_group("sim_results")
            sim_results.create_dataset("image", data=img, dtype="i1")
            sim_results.create_dataset("num_mc_steps", data=m_all[rep], dtype="i")

            region_props = sim_results.create_group("region_props")
            region_props.create_dataset("euler_number", data=region["euler_number"])
            region_props.create_dataset("normalised_euler_number", data=region["euler_number"] / np.sum(img == 2))
            region_props.create_dataset("perimeter", data=region["perimeter"], dtype="f")
            region_props.create_dataset("eccentricity", data=region["eccentricity"], dtype="f")

            master_file.close()

            if (cat == "none") and (self.generate_mode == "make_dataset"):
                os.remove(f"{self._file_base}--{self.sweep_cnt}.h5")

            if self.sftp_when_done:
                self.network_rabanis()

            self.sweep_cnt += 1

    @staticmethod
    def calculate_stats(img, image_res, substrate_num=0, liquid_num=1, nano_num=2):
        # Region Properties
        region = (measure.regionprops((img != 0) + 1)[0])

        # Broadly estimate category
        if int(mode(img, axis=None).mode) == liquid_num:
            if np.sum(img == substrate_num) / image_res ** 2 >= 0.02:
                # Hole if dominant category is water and also has an amount of substrate
                cat = "hole"
            else:
                # Liquid if dominant category is water (==1)
                cat = "liquid"
        elif -0.00025 <= region["euler_number"] / np.sum(img == nano_num):
            # Cell/Worm if starting to form
            cat = "cellular"
        elif -0.01 <= region["euler_number"] / np.sum(img == nano_num) < -0.001:
            # Labyrinth
            cat = "labyrinth"
        elif region["euler_number"] / np.sum(img == nano_num) <= -0.03:
            # Island
            cat = "island"
        else:
            cat = "none"

        return region, cat

    def network_rabanis(self):
        if not self.ssh:
            self.setup_ssh()
        for file in os.listdir(self._dir_base):
            self.sftp.put(file, f"/home/mltest1/tmp/pycharm_project_883/{self._dir_base}/{file}")
            os.remove(f"{self._dir_base}/{file}")


if __name__ == '__main__':
    root_dir = "Data/Simulated_Images"

    total_image_reps = 1

    parameters = {"kT": [0.07, 0.4],
                  "mu": [2.35, 3.3],
                  "MR": 1,
                  "C": 0.3,
                  "e_nl": 1.5,
                  "e_nn": 2,
                  "L": [64, 200]}

    axis_res = {"kT": 25,
                "mu": 25,
                "L": 5}

    rabani_sweeper = RabaniSweeper(root_dir=root_dir, generate_mode="make_dataset")
    rabani_sweeper.call_rabani_sweep(params=parameters,
                                     axis_steps=axis_res,
                                     image_reps=total_image_reps)
