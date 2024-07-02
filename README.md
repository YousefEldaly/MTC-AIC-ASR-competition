# README

## International Competition of the Military Technical College (AI Competition)

This README is based on the requirements of the “International Competition of the 
Military Technical College” AI competition. For more information, visit the following links:
- [MTC-AIC Overview](https://aic.conferences.ekb.eg/)
- [Kaggle Competition Page](https://www.kaggle.com/competitions/mct-aic-2/overview)

The provided dataset comprises 100 hours of clean and noisy audio recordings in the 
Egyptian dialect. The goal is to minimize the Word Error Rate (WER) using a suitable model. 
Our team, Hear to Win, achieved a WER of 26.17 using a small Jasper model from the NeMo toolkit, 
ranking 12th out of 150+ registered teams.

## Steps Taken

1. **Determining the Training Platform**
   - **Google Colab**: Efficient GPU but limited in access time.
   - **AzureML Compute Instances and Clusters**: Offers $100 of free access to different CPUs for students. 
     However, using a custom virtual environment with Python==3.10 resulted in the notebook reverting to Python 3.19.4, 
     which is incompatible with the NeMo toolkit models requiring Python>=3.10.
   - **Kaggle**: Given the limited time (30 hours per user * 3 team members) and total control of the environment, 
     Kaggle was selected as our training platform.

2. **Data Exploration and Cleaning**
   - The dataset contained some null values and HTML tags, which were cleaned.

3. **Formatting the Data**
   - The data was formatted into the JSON format required by NeMo: 80% for training, 10% for testing, 
     and 10% for validation.

4. **Model Training**
   - Ran a Conformer Transducer model with simple configurations. The setup can be found 
     [here](https://github.com/YousefEldaly/MTC-AIC-ASR-competition/blob/main/data-setup-clean-train-trial.ipynb).

5. **Training Various Models**
   - Trained different models including CTC, Transducer, Jasper, and Hybrid-Transducer-CTC with various model dimensions, 
     encoders, decoders, and tokenizers.
   - Adjusted configurations to achieve the best WER. Configurations for each model are documented in their 
     respective notebooks.

## Conclusion

- Large model dimensions resulted in worse WER. Models with an encoder dimension of 512 or higher often produced empty predictions for most audio files, regardless of the number of training epochs.
- Recommended using small or medium models with encoder dimensions of 176 or 256 respectively.
- Using SpecAugment improved the model's ability to handle noisy audio tracks.
- Training for 15-60 epochs led to a generalized model, while more epochs caused overfitting.

