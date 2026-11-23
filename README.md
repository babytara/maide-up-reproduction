# # MAiDE-up Reproduction

CSCI 3052U Independent Activity (Hacker / Implementer role)

Paper: [Ignat, Xu & Mihalcea (2025). *MAiDE-up: Multilingual Deception Detection of AI-generated Hotel Reviews*. Findings of NAACL 2025.](https://aclanthology.org/2025.findings-naacl.88/)

Dataset: Authors' public repo: [MichiganNLP/multilingual_reviews_deception](https://github.com/MichiganNLP/multilingual_reviews_deception)

---
## Repo structure 
```
maide-up-reproduction/
├── 01_Data_and_Baseline.ipynb   # data checks, cleaning, split, TF-IDF + Naive Bayes / Random Forest
├── 02_xlmr.ipynb                # XLM-RoBERTa fine-tuning: full run, few-shot run, per-language results
├── requirements.txt
└── README.md
```

## Pipeline

1. **Load** `all_data.csv` from the authors' repo.
2. **Data checks** `Prompt_Language` contains 100% of fake reviews and 0% of real ones, so I dropped it as a feature. The real reviews had upside or downside half blank much more often than fake reviews. I tested this separately in step 6.
3. **Data cleaning** I combined the upside and downside halves into one column - `text`. Only `text`, `label`, `Review_Language` and `Sentiment` were kept. Hotel and city names were dropped.
4. **Data split** 80/20 split was used into train and test, stratified by label and language, the seed is 42, that resulted in 15,988 train and 3,997 test each. Both notebooks used the same split to rebuild.
5. **TF-IDF baselines** (Notebook 01): Word-level and Character-level TF-IDF with Multinomial Naive Bayes and Random Forest were used.
6. **Leakage test** (Notebook 01): Logistic Regression trained only on "is the upside / downside half blank?".
7. **XLM-RoBERTa** (Notebook 02): 10% of the training set was held out for validation, the split was 14,389 / 1,599, I fine-tuned for 5 epochs, then selected the best epoch and picked by validation accuracy, evaluated on the 3,997 test reviews.
8. **Few-shot** (Notebook 02): A fresh XLM-R was trained on 200 reviews, 10 real + 10 fake for each language, tested on the same test set.
9. Per-language accuracy for word RF, character RF and XLM-R.

## How to reproduce results

1. Open a new notebook in Google Colab
2. Upload the notebooks to Google Colab (File → Upload notebook)
3. Choose the runtime for the notebook:
   - `01_data_and_baseline.ipynb`: CPU is sufficient. Takes a few minutes to rebuild.
   - `02_xlmr.ipynb`: Change the runtime type to **T4 GPU**. The full training run takes about 25 minutes.
4. Run all cells from top to bottom. The first code cell clones the authors' data from their repository.

**To run locally on your computer:**
`pip install -r requirements.txt`, then open the notebooks in Jupyter/a code IDE. Notebook 02 requires a CUDA GPU.

## Settings Used

| Setting | Value |
|---|---|
| Seed | 42 (`random`, `numpy`, `torch`, `transformers.set_seed`) |
| TF-IDF (word) | `max_features=50000` |
| TF-IDF (character) | `analyzer="char_wb"`, `ngram_range=(2, 4)`, `max_features=50000` |
| Random Forest | `n_estimators=200` |
| XLM-R model | `xlm-roberta-base` |
| Learning rate / batch size / epochs | 2e-5 / 8 / 5 (same as the paper) |
| Max length | 256 tokens (95th percentile of review length was 149; 99% of reviews fit) |
| Best epoch | chosen by validation accuracy (`load_best_model_at_end=True`) |
| Mixed precision | `fp16=True` on a T4 GPU |

## Results

Test set: 3,997 reviews. Paper numbers are from Table 4 of the paper.

| Model | Paper accuracy | My accuracy | Δ (pp) | Paper F1 | My F1 |
|---|---:|---:|---:|---:|---:|
| Naive Bayes, word TF-IDF | 84.5 | 86.46 | +1.96 | 83.9 | 87.90 |
| Naive Bayes, character TF-IDF | – | 89.92 | – | – | 90.53 |
| Random Forest, word TF-IDF | 87.7 | 91.09 | +3.39 | 86.9 | 91.01 |
| Random Forest, character TF-IDF | – | 91.27 | – | – | 91.29 |
| XLM-R, full training set | 94.8 | 97.32 | +2.52 | 94.9 | 97.35 |
| XLM-R, few-shot (200 reviews) | 76.6 | 87.24 | +10.64 | 80.1 | 87.99 |
| Blank-halves only (Logistic Regression) | – | 60.07 | – | – | – |

### Accuracy by language (%)

| Language | Word RF | Character RF | XLM-R |
|---|---:|---:|---:|
| Chinese | 80.35 | 90.93 | 97.73 |
| English | 90.25 | 87.00 | 97.75 |
| French | 93.50 | 90.75 | 98.00 |
| German | 90.25 | 88.75 | 97.25 |
| Italian | 94.00 | 92.25 | 96.50 |
| Korean | 89.75 | 92.25 | 97.25 |
| Romanian | 91.75 | 90.00 | 96.25 |
| Russian | 93.25 | 93.75 | 98.25 |
| Spanish | 94.50 | 92.75 | 97.25 |
| Turkish | 93.25 | 94.25 | 97.00 |

Each language has about 400 test reviews, so one review is weights about 0.25 percentage points.

## Analysis

- All of my results exceed the paper's, however the setups were different, so this is not mean its a better method.
- Chinese was the weakest language for word TF-IDF at about 80.35%. Word TF-IDF splits based on spaces, and Chinese doesn't use spaces between words. Character n-grams raised it accuracy to 90.93%, and XLM-R's subword tokenizer raised it to about 97.73%.
- XLM-R may overfit after its best epoch run. Training loss consistently fell (0.012 at epoch 5) while validation loss rose and validation accuracy decreased from 96.2% to 94.8% between epochs 4 and 5. I think selecting the best epoch on validation data matters.
- The dataset includes a non-linguistic signal. If it simply checking whether one half of a review is blank yields 60.07% accuracy, compared with a 50% chance baseline.

## Differences from the paper

- I used Multinomial NB instead of Gaussian NB.
- 200 trees were instead of the default, 100.
- I only used TF-IDF text features. The paper used analytic, descriptive and readability scores as well.
- The split was stratified by label and language. The authors' notebook stratifies by label only.
- The max lenght was 256 tokens, I selected this  from the data. The paper did not report its value.
- The paper tests few-shot on 19,800 reviews. I used my 3,997-review test set. I also picked the best few-shot epoch using 1,599 validation reviews, which is a lot of labeled data for a few-shot setting.
- My library versions may differ from the authors'.

## Notes

- I fixed all random seeds to 42 and used the same train/test split in both notebooks.
- The XLM-R model is still not perfectly reproducible on a GPU. Running the exact same code twice produced 97.35% accuracy on one run and 97.32% on another. This happens because mixed-precision training can introduce small numerical differences. Using full_determinism=True would make the results fully repeatable, but training would take longer. The results reported here are from the second run, which is the version saved in the notebook.
- I only evaluated a single seed. Testing multiple seeds would provide a better picture of how consistent the results are and how meaningful the gap to the paper really is.
- All package versions are listed in requirements.txt, and training was done on a free Google Colab T4 GPU.

Versions are from the Google Colab environment used for this project (November 2026). Colab used the CUDA 13.0 build of torch (2.14.1+cu130).

## Citation

```bibtex
@inproceedings{ignat-etal-2025-maide,
    title = "{MA}i{DE}-up: Multilingual Deception Detection of {AI}-generated Hotel Reviews",
    author = "Ignat, Oana and Xu, Xiaomeng and Mihalcea, Rada",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2025",
    year = "2025",
    pages = "1636--1653",
    url = "https://aclanthology.org/2025.findings-naacl.88/"
}
```
