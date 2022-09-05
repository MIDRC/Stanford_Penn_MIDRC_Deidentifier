# Stanford & Penn Deidentifier

This deidentifier was developped on radiology reports to automate the de-identification process, before using the reports in a research project or sharing them with other teams or institutions. It was also tested on other types of clinical notes and demonstrated high performance, as underlined in the associated publication.

This project was made possible by MIDRC and part of **MIDRC CRP 1 - Natural language processing of radiology reports for COVID-19** (https://www.midrc.org/midrc-collaborating-research-projects/project-one-crp1).

**Development Team**: Pierre Chambon (Stanford University), Tessa S. Cook (Penn University), Curtis P. Langlotz (Stanford University).

# Getting the model running

```console
foo@bar:~$ git clone https://github.com/MIDRC/Stanford_Penn_Deidentifier.git
foo@bar:~$ cd Stanford_Penn_Deidentifier
```

Then, make sure you are in a Python virtualenv or conda environment where torch is already installed: otherwise, follow the instructions of https://pytorch.org/.
Once torch is installed, you can run the command:

```console
foo@bar:~$ pip install -r requirements.txt
```

You should be set up to make the model work. To run the model on your own reports, you need to put the reports, represented as a python string array, in a .npy file. The deidentifier will put the deidentified reports in another .npy file, along a .csv file with more info that can be useful for a human review of the deidentified reports (prior to sharing or public release).

To run the model, you only need the command:

```console
foo@bar:~$ python main.py [-h] --device_list DEVICE_LIST [DEVICE_LIST ...]
               [--num_workers NUM_WORKERS] [--num_cpu_processes NUM_CPU_PROCESSES]
               [--batch_size BATCH_SIZE] --input_file_path INPUT_FILE_PATH
               --output_file_path OUTPUT_FILE_PATH
               [--hospital_list HOSPITAL_LIST [HOSPITAL_LIST ...]]
               [--vendor_list VENDOR_LIST [VENDOR_LIST ...]]
```

As an example, you could run:

```console
foo@bar:~$ python main.py --input_file_path ./reports_stanford.npy --output_file_path ./reports_stanford_deidentified.npy --device cuda:0 cuda:1 cuda:2 --hospital_list stanford washington
```

References
---
1)  For information on MIDRC GitHub documentation and best practices, please see https://midrc.atlassian.net/wiki/spaces/COMMITTEES/pages/672497665/MIDRC+GitHub+Best+Practices
2)	U-Net: Convolutional Networks for Biomedical Image Segmentation https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
3)  Laboratory and University Name that develops the algorithm.
