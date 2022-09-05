import argparse
import numpy as np
import torch
from hide_in_plain_sight import hide_in_plain_sight
from deidentifier_model import deidentifier_model
from multiprocessing import Process
import time
import random
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Deidentify radiology reports")
    parser.add_argument(
        "--device_list",
        nargs="+",
        help="Devices to run the transformer(s) model on, can provide several spaced device names to scale. Must be one of cpu, mps, cuda, or cuda:device_number",
        required=True,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for the transformer",
    )
    parser.add_argument(
        "--num_cpu_processes",
        type=int,
        default=1,
        help="If running on cpu, can split the deidentification between several processes",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference with the transformer model",
    )
    parser.add_argument(
        "--input_file_path",
        type=str,
        help="Path to input file, must be .npy",
        required=True,
    )
    parser.add_argument(
        "--output_file_path",
        type=str,
        help="Path to output file, must be .npy",
        required=True,
    )
    parser.add_argument(
        "--hospital_list",
        nargs="+",
        help="Hospitals to be parsed based on lower-case matching",
        required=False,
    )
    parser.add_argument(
        "--vendor_list",
        nargs="+",
        help="Vendors to be parsed based on lower-case matching",
        required=False,
    )

    return parser.parse_args()


def check_args_validity(args):
    assert args.batch_size >= 0
    assert args.input_file_path[-4:] == ".npy"
    assert args.output_file_path[-4:] == ".npy"

    assert len(args.device_list) > 0
    for device in args.device_list:
        if device not in ["cpu", "mps", "cuda"]:
            assert device[:5] == "cuda:"
            assert device[5:].isnumeric()


def deidentifier_model_and_hide_in_plain_sight(
    file_seed, device, num_workers, batch_size, hospital_list, vendor_list
):
    deidentifier_model(
        file_seed, device, num_workers, batch_size, hospital_list, vendor_list
    )
    hide_in_plain_sight(file_seed)


def generate_output_files(file_seed_list, output_file_path):
    reports = []
    labeled_reports = []
    deidentified_reports = []
    phi_lengths = []

    for file_seed in file_seed_list:
        with open("original_reports" + file_seed + ".npy", "rb") as f:
            reports.extend(np.load(f, allow_pickle=True))
        os.remove("original_reports" + file_seed + ".npy")

        with open("labeled_reports" + file_seed + ".npy", "rb") as f:
            labeled_reports.extend(np.load(f, allow_pickle=True))
        os.remove("labeled_reports" + file_seed + ".npy")

        with open("deidentified_reports" + file_seed + ".npy", "rb") as f:
            deidentified_reports.extend(np.load(f, allow_pickle=True))
        os.remove("deidentified_reports" + file_seed + ".npy")

        with open("phi_lengths" + file_seed + ".npy", "rb") as f:
            phi_lengths.extend(np.load(f, allow_pickle=True))
        os.remove("phi_lengths" + file_seed + ".npy")

    with open(output_file_path, "wb") as f:
        np.save(f, deidentified_reports)

    df_for_review = pd.DataFrame(
        [reports, labeled_reports, deidentified_reports, phi_lengths]
    ).transpose()
    df_for_review.rename(
        columns={
            0: "original_reports",
            1: "labeled_reports",
            2: "deidentified_reports",
            3: "phi_lengths",
        },
        inplace=True,
    )
    df_for_review.to_csv("deidentification_details_for_review.csv", index=False)


def main(args):
    # Load the reports
    start = time.time()

    with open(args.input_file_path, "rb") as f:
        reports = np.load(f, allow_pickle=True)

    print("Processing", str(len(reports)), "reports")

    device_list = (
        args.device_list
        if (len(args.device_list) > 1 or args.device_list[0] != "cpu")
        else ["cpu" for _ in range(args.num_cpu_processes)]
    )

    device_list = [torch.device(device) for device in device_list]
    file_seed_list = []
    number_reports_per_file = len(reports) // len(device_list)

    # Prepare files to be processed in each separate process
    for i, device in enumerate(device_list):
        file_seed = (
            device.type
            + "device"
            + str(i)
            + "device"
            + str(random.randint(2394492340, 23944923402394492340))
        )
        file_seed_list.append(file_seed)

        with open(
            "original_reports" + file_seed + ".npy",
            "wb",
        ) as f:
            np.save(
                f,
                reports[
                    i
                    * number_reports_per_file : (
                        ((i + 1) * number_reports_per_file)
                        if i + 1 < len(device_list)
                        else len(reports)
                    )
                ],
            )

    processes = []

    for file_seed, device in zip(file_seed_list, device_list):
        p = Process(
            target=deidentifier_model_and_hide_in_plain_sight,
            args=(
                file_seed,
                device,
                args.num_workers,
                args.batch_size,
                args.hospital_list if args.hospital_list is not None else [],
                args.vendor_list if args.vendor_list is not None else [],
            ),
        )
        p.start()
        processes.append(p)
        # Running without parallelization
        # deidentifier_model_and_hide_in_plain_sight(
        # file_seed,
        # device,
        # args.num_workers,
        # args.batch_size,
        # args.hospital_list,
        # args.vendor_list,
        # )
        pass

    for p in processes:
        p.join()

    generate_output_files(file_seed_list, args.output_file_path)

    print("Ended execution, took ", time.time() - start, " seconds")


if __name__ == "__main__":
    args = parse_args()
    check_args_validity(args)
    main(args)
