import json
import os
import platform
from datetime import datetime
from itertools import product

import h5py
import numpy as np
import paramiko
from scipy.stats import mode
from skimage import measure

from rabani import _run_rabani_sweep


class RabaniSweeper:
    def __init__(self, root_dir, sftp_when_done=False, generate_mode='visualise'):
        self.system_name = platform.node()
        self.root_dir = root_dir

        self.generate_mode = generate_mode
        self.sftp_when_done = sftp_when_done
        self.ssh = None
        self.sftp = None

        self.start_datetime = datetime.now()
        self.start_date = self.start_datetime.strftime("%Y-%m-%d")
        self.start_time = self.start_datetime.strftime("%H-%M")
        self.end_datetime = None
        self.params = None

        self.sweep_cnt = 1

    def setup_ssh(self):
        with open("details.json", 'r') as f:
            details = json.load(f)

        self.ssh = paramiko.SSHClient()
        self.ssh.load_system_host_keys()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(details["ip_addr"], username=details["user"], password=details["pass"])
        self.sftp = self.ssh.open_sftp()

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
        L = get_linspace_ranges(params["L"], axis_steps)

        current_time = self.start_datetime.strftime("%H:%M:%S")
        print(f"{current_time} - Beginning generation of {axis_res * axis_res * image_reps} rabanis")

        self.params = np.array(
            list(product(kT_linspace, mu_linspace, MR_linspace, C_linspace, e_nl_linspace, e_nn_linspace, L)))

        for image_rep in range(image_reps):
            imgs, m_all = _run_rabani_sweep(self.params)
            imgs = np.swapaxes(imgs, 0, 2)
            self.save_rabanis(imgs, m_all)

            now = datetime.now().strftime("%H:%M:%S")
            print(
                f"{now} - Successfully completed block {image_rep + 1} of {image_reps} ({axis_res * axis_res} rabanis)")

        self.end_datetime = datetime.now()

    def make_storage_folder(self, dir):
        if not os.path.isdir(dir):
            os.makedirs(dir)

    def save_rabanis(self, imgs, m_all):
        self.make_storage_folder(f"{self.root_dir}/{self.start_date}/{self.start_time}")
        for rep, img in enumerate(imgs):
            master_file = h5py.File(
                f"{self.root_dir}/{self.start_date}/{self.start_time}/rabanis--{platform.node()}--{self.start_date}--{self.start_time}--{self.sweep_cnt}.h5",
                "a")

            region, cat = self.calculate_stats(img)

            master_file.attrs["kT"] = self.params[rep, 0]
            master_file.attrs["mu"] = self.params[rep, 1]
            master_file.attrs["MR"] = self.params[rep, 2]
            master_file.attrs["C"] = self.params[rep, 3]
            master_file.attrs["e_nl"] = self.params[rep, 4]
            master_file.attrs["e_nn"] = self.params[rep, 5]
            master_file.attrs["L"] = self.params[rep, 6]
            master_file.attrs["category"] = cat

            sim_results = master_file.create_group("sim_results")
            sim_results.create_dataset("image", data=img, dtype="i1")
            sim_results.create_dataset("num_mc_steps", data=m_all[rep], dtype="i")

            region_props = sim_results.create_group("region_props")
            region_props.create_dataset("euler_number", data=region["euler_number"])
            region_props.create_dataset("normalised_euler_number", data=region["euler_number"]/np.sum(img == 2))
            region_props.create_dataset("perimeter", data=region["perimeter"], dtype="f")
            region_props.create_dataset("eccentricity", data=region["eccentricity"], dtype="f")

            if cat is not "none":
                if self.sftp_when_done:
                    self.network_rabanis()
            elif self.generate_mode is "make_dataset":
                os.remove(f"{self.root_dir}/{self.start_date}/{self.start_time}/rabanis--{platform.node()}--{self.start_date}--{self.start_time}--{self.sweep_cnt}.h5")

            self.sweep_cnt += 1

    def calculate_stats(self, img):
        # Region Properties
        region = (measure.regionprops((img != 0) + 1)[0])

        # Broadly estimate category
        if int(mode(img, axis=None).mode) == 1:
            if np.sum(img == 0) / self.params[0, 6] ** 2 >= 0.02:
                # Hole if dominant category is water and also has an amount of substrate
                cat = "hole"
            else:
                # Liquid if dominant category is water (==1)
                cat = "liquid"
        elif -0.015 <= region["euler_number"]/np.sum(img == 2) <= 0:
            # Cell/Worm if starting to form
            cat = "cellular"
        elif -0.03 <= region["euler_number"]/np.sum(img == 2) < -0.015:
            # Labyrinth
            cat = "labyrinth"
        elif region["euler_number"]/np.sum(img == 2) <= -0.05:
            # Island
            cat = "island"
        else:
            cat = "none"

        return region, cat

    def network_rabanis(self):
        if not self.ssh:
            self.setup_ssh()
        self.sftp.put(
            f"{self.root_dir}/{self.start_date}/{self.start_time}/rabanis--{platform.node()}--{self.start_date}--{self.start_time}--{self.sweep_cnt - 1}.h5",
            f"/home/mltest1/tmp/pycharm_project_883/Images/ImageDump/rabanis--{platform.node()}--{self.start_date}--{self.start_time}--{self.sweep_cnt - 1}.h5")
        os.remove(
            f"{self.root_dir}/{self.start_date}/{self.start_time}/rabanis--{platform.node()}--{self.start_date}--{self.start_time}--{self.sweep_cnt}.h5")


if __name__ == '__main__':
    root_dir = "Images"
    total_image_reps = 1
    axis_res = 40

    kT_range = [0.01, 0.35]
    mu_range = [2.35, 3.47]
    MR_range = 1
    C_range = 0.3
    e_nl_range = 1.5
    e_nn_range = 2
    L = 128

    rabani_sweeper = RabaniSweeper(root_dir=root_dir, generate_mode="make_dataset")
    rabani_sweeper.call_rabani_sweep(params={"kT": kT_range,
                                             "mu": mu_range,
                                             "MR": MR_range,
                                             "C": C_range,
                                             "e_nl": e_nl_range,
                                             "e_nn": e_nn_range,
                                             "L": L},
                                     axis_steps=axis_res, image_reps=total_image_reps)
