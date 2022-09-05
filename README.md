# Stanford & Penn Deidentifier

:hugs: Try out our model demo at https://huggingface.co/StanfordAIMI/stanford-deidentifier-base !

This deidentifier was developped on radiology reports to automate the de-identification process, before using the reports in a research project or sharing them with other teams or institutions. It was also tested on other types of clinical notes and demonstrated high performance, as underlined in the associated publication.

This project was made possible by MIDRC and part of **MIDRC CRP 1 - Natural language processing of radiology reports for COVID-19** (https://www.midrc.org/midrc-collaborating-research-projects/project-one-crp1).

**Development Team**: Pierre Chambon (Stanford University), Tessa S. Cook (Penn University), Curtis P. Langlotz (Stanford University).

# Getting the model running

```bat
foo@bar:~$ git clone https://github.com/MIDRC/Stanford_Penn_Deidentifier.git
foo@bar:~$ cd Stanford_Penn_Deidentifier
```

Then, make sure you are in a Python virtualenv or conda environment where torch is already installed: otherwise, follow the instructions of https://pytorch.org/.
Once torch is installed, you can run the command:

```bat
foo@bar:~$ pip install -r requirements.txt
```

You should be set up to make the model work. To run the model on your own reports, you need to put the reports, represented as a python string array, in a .npy file. The deidentifier will put the deidentified reports in another .npy file, along a .csv file with more info that can be useful for a human review of the deidentified reports (prior to sharing or public release).

To run the model, you only need the command:

```bat
foo@bar:~$ python main.py [-h] --device_list DEVICE_LIST [DEVICE_LIST ...]
               [--num_workers NUM_WORKERS] [--num_cpu_processes NUM_CPU_PROCESSES]
               [--batch_size BATCH_SIZE] --input_file_path INPUT_FILE_PATH
               --output_file_path OUTPUT_FILE_PATH
               [--hospital_list HOSPITAL_LIST [HOSPITAL_LIST ...]]
               [--vendor_list VENDOR_LIST [VENDOR_LIST ...]]
```

As an example, you could run:

```bat
foo@bar:~$ python main.py --input_file_path ./reports_stanford.npy --output_file_path ./reports_stanford_deidentified.npy --device cuda:0 cuda:1 cuda:2 --hospital_list stanford washington
```

# Having good synthetic PHI

The hide-in-plain-sight algorithm generates synthetic PHI in-place of the original and detected PHI. To do so, it relies on data sets of PHI examples, for each category. This repo contains the data files for each category but with very limited data in them. You can either modify them manually, and insert whatever examples of names or hospitals you wish, or rely on the following online resources to get good data sets of synthetic PHI:

- Surnames. Go to https://www.census.gov/topics/population/genealogy/data/2000_surnames.html and download "File B: Surnames Occurring 100 or more times". Then rename the file as "Common_Surnames_Census_2000.csv" and replace the existing file of this repo, already named "Common_Surnames_Census_2000.csv", with this new file. 

- Firstnames. Go to https://data.world/len/us-first-names-database/workspace/file?filename=SSA_Names_DB.xlsx and download the file "SSA_Names_DB.xlsx". Then replace the existing file of this repo, already named "SSA_Names_DB.xlsx", with this new file. 

- Vendors/Companies. You can find good examples at 'https://golden.com/list-of-healthcare-companies/’ and then insert them in the file "companies.txt".

- Hospitals. You can find good examples at 'https://www.hospitalsafetygrade.org/all-hospitals’ and then insert them in the file "hospitals.txt".

- Universities. You can find good examples at 'https://en.wikipedia.org/wiki/Lists_of_American_universities_and_colleges’ and then insert them in the file "universities.txt".

# References
---
Manuscript currently in proceedings.
