# Text-Based Ideal Points Model (Code and Data) for US House, Sessions 115 and 116 (2017-2021).

## This data and code uses TBIP to estimate ideal point values for legislators from their texts, and also estimates vote-based ideal points with the same framework to enable comparison of ideological positionining and expression of members of Congress across three venues: _Floor Speeches_, _Twitter Tweets_, and _Roll-Call Votes_. 

**First, please read the README at the original TBIP repo (https://github.com/keyonvafa/tbip) to get the required overview, and install the required libraries (pip install -r requirements.txt) in your environment. It is important to understand the data files and basic structure from the original TBIP code as this directory directly builds from that! Please cite the TBIP paper by Vafa et al. if using this software: Vafa, K., Naidu, S., & Blei, D. (2020, July). Text-Based Ideal Points. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (pp. 5345-5357).** 

## Data
---

There are three datasets provided, one for each of the venues (audiences) studied in this work:  
  
  1. **floor_speeches_congs_115_116**
  2. **tweets_cong_115_116**
  3. **congs_115-116_votes**

Below, we highlight the _pipeline used for each of these datasets to convert raw text or vote data to ideal point estimates_. The ideal point estimates along with a host of information about the legislators are then combined and presented in our main file used for analyses conducted in the paper (`../legislator_info_and_tbip_congresses_115_and_116.csv`); **for combining all results and creating that file, please view the documented code in the main project directory (`../combine_and_create_main_file_for_conducting_research.ipynb`)**. 

NOTE: For source/reference information regarding each raw data that serves as the starting point for ideal point estimation process, please view the README in the corresponding data/ subdirectory. 

## Process for deriving ideal point estimates as well as various ideological topic modeling estimates from raw data: 


### Floor Speeches
---

**Step 1:** 

Input: Raw data file for floor speeches derived from the Congressional Record: `data/floor_speeches_congs_115_116/raw_original_data_floor_speeches_house.csv` 

Output: Processed files in `data/floor_speeches_congs_115_116/clean/`

Process: Run the script: `data/floor_speeches_congs_115_116/preprocess_floor_speeches_and_convert_to_bag_of_words.ipynb` 


**Step 2:** 

Input: Processed data files in `data/floor_speeches_congs_115_116/clean/`

Output: Processed floor speech data after removing procedural speeches in: `data/floor_speeches_congs_115_116/clean_removing_procedural/`

NOTE: This data cleaning step is particular to floor speeches, since many floor speeches can be low on content and high on procedural legislative jargon: both the logic of the code and the intuition behind removing such speeches comes from: **Card, Dallas, Serina Chang, Chris Becker, Julia Mendelsohn, Rob Voigt, Leah Boustan, Ran Abramitzky, and Dan Jurafsky. "Computational analysis of 140 years of US political speeches reveals more positive but increasingly polarized framing of immigration." Proceedings of the National Academy of Sciences 119, no. 31 (2022): e2120510119.** 

Process: 

1. Run the following script: `python data/floor_speeches_congs_115_116/filter_procedural.py --input_fpath clean/raw_documents.txt --output_fpath raw_documents_without_procedural.txt`

2. Then, using the above creating txt file, run `data/floor_speeches_congs_115_116/preprocessing_raw_speeches_after_procedural_speech_removal.ipynb` to create the files in `data/floor_speeches_congs_115_116/clean_removing_procedural/`

3. Finally, creating a json file for the vocab (useful for the next step) by running the python script: `data/floor_speeches_congs_115_116/clean_removing_procedural/vocab_txt_to_json.py`


**Step 3.1 (can be run in parallel with 3.2):** 

Input: Processed floor speech data after removing procedural speeches in: `data/floor_speeches_congs_115_116/clean_removing_procedural/`

Output: Poisson factorization topic modeling output results stored in 10 subdirectories of the form: `data/floor_speeches_congs_115_116/pf-fits-removed-procedural-speeches-k50-seed*`

Process: Run - `poisson_scripts/floor_speeches_congs_115_116.sh`

NOTE: Poisson factorization is run ten times with different random seeds to get an expected mean value in order to scale MALLET topic modeling output: this scaling is needed in order to use MALLET topic modeling results as input to text-based ideal point estimation. 

**Step 3.2 (can be run in parallel with 3.1):** 

Input: Processed floor speech data after removing procedural speeches in: `data/floor_speeches_congs_115_116/clean_removing_procedural/`

Output: MALLET topic modeling output files in `data/floor_speeches_congs_115_116/mallet_fits_removed_procedural_speeches/`

Process:

1. Get the soup-nuts package to run MALLET topic model from: https://github.com/ahoho/topics (follow the instructions); and cite the following: 

`
@inproceedings{hoyle-etal-2021-automated,
    title = "Is Automated Topic Evaluation Broken? The Incoherence of Coherence",
    author = "Hoyle, Alexander Miserlis  and
      Goel, Pranav  and
      Hian-Cheong, Andrew and
      Peskov, Denis and
      Boyd-Graber, Jordan and
      Resnik, Philip",
    booktitle = "Advances in Neural Information Processing Systems",
    year = "2021",
    url = "https://arxiv.org/abs/2107.02173",
}
`

2. Adjusting file paths accordingly, run: `python soup_nuts/models/gensim/lda.py --input_dir data/floor_speeches_congs_115_116/clean_removing_procedural/ --model mallet --output_dir /workspace/pranav/tbip/data/floor_speeches_congs_115_116/mallet_fits_removed_procedural_speeches --train_path counts.npz --eval_path counts.npz --vocab_path vocabulary.json --num_topics 50 --optimize_interval 10 --workers 8`


**Step 4:**

Input: MALLET topic modeling output files in `data/floor_speeches_congs_115_116/mallet_fits_removed_procedural_speeches/`

Output: Scaled MALLET topic modeling output files for use in subsequent ideal point estimation, also stored in `data/floor_speeches_congs_115_116/mallet_fits_removed_procedural_speeches/`

Process: Run the following script: `python setup/scale_mallet_output_using_poisson_factorization_runs.py --base_dir data/floor_speeches_congs_115_116/ --glob_pattern "pf-fits-removed-procedural-speeches-k50-seed*" --input_mallet_dir data/floor_speeches_congs_115_116/mallet_fits_removed_procedural_speeches/ --beta_fname beta.npy --theta_fname doctopics.txt`


**Step 5:**

Input: `data/floor_speeches_congs_115_116/clean_removing_procedural/` and `data/floor_speeches_congs_115_116/mallet_fits_removed_procedural_speeches/`

Output: `data/floor_speeches_congs_115_116/tbip-pytorch-fits-og-rem-procedural-speeches-k50-init-mallet/`

Process: Run the script: `tbip_scripts/floor_speeches_congs_115_116.sh`

NOTE: We highly recommending running the above script using a GPU device rather than on CPU. 


**Step 6:**

Input: Host of subdirectories present in `data/floor_speeches_congs_115_116/`

Output: Files stored in `../speeches_results/` 

Process: Running the code in the notebook: `analysis/analyze_floor_speeches_ideal_points.ipynb`
