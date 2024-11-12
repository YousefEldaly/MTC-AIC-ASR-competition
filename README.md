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

### Conformer Model Diagram

![Conformer Architecture](https://github.com/YousefEldaly/MTC-AIC-ASR-competition/blob/main/Conformer-encoder-architecture.png)

_Figure 1: Diagram of the Conformer Architecture_

### Conformer Architecture Breakdown

The Conformer model has several key components:

1. **Preprocessing**: Audio data is first preprocessed by converting raw waveform inputs into spectrogram features, specifically Mel spectrograms. This transformation provides a frequency-based representation that is more effective for ASR tasks.

2. **Augmentation**: Data augmentation techniques, such as SpecAugment, are applied to make the model more robust to variations. SpecAugment randomly masks portions of the spectrogram along time and frequency dimensions to improve generalization.

3. **Encoder**: The Conformer encoder consists of convolutional layers for capturing local dependencies, and self-attention layers for capturing global dependencies. Each encoder layer is structured with feed-forward, self-attention, and convolution blocks.

4. **CTC/Transducer Decoding**: Depending on the model type, the decoder uses either a Connectionist Temporal Classification (CTC) or a Transducer setup:

   - **CTC** aligns input sequences with output labels for frame-based prediction.
   - **Transducer** models (used in Transducer and Hybrid architectures) combine encoder and decoder outputs for sequence prediction.

5. **Output Layer**: The final layer maps the encoded features to character or sub-word labels, depending on the configuration.

## Models Overview
| Model Name                       | Type         | Label Type | Key Components              | Tokens   | WER (epochs)               | Notes                                |
| -------------------------------- | ------------ | ---------- | --------------------------- | -------- | --------------------- | ------------------------------------ |
| Conformer-Hybrid-Transducer-CTC  | Hybrid       | Character  | Encoder: Conformer          | 675,790  | Not converged (100)   | Combines Transducer & CTC            |
| Conformer-CTC-Char               | CTC          | Character  | Encoder: Conformer          | 675,790  | 13.2 (40)             | Configured from scratch              |
| Conformer-Transducer-Char        | Transducer   | Character  | Encoder: Conformer, Decoder | 675,790  | Not converged (80)    | RNNTDecoder & Joint reduction        |
| Jasper                           | CTC          | Character  | Deep CNN Layers             | N/A      | 24.4 (30)             | Deep CNN for ASR                     |

Note: Some of these model are in [MTC-AIC phase2 repo](https://github.com/YousefEldaly/MTC-AIC-ASR-Competition-phase2)

## Re-Test the Submitted Model

To re-test the submitted model using the provided notebook, follow these steps:

1. **Download the Notebook:**

   - Go to [inference-script.ipynb](https://github.com/YousefEldaly/MTC-AIC-ASR-competition/blob/main/submitted_model/inference-script.ipynb).
   - Click on the "Raw" button to download the notebook file.

2. **Upload to Kaggle:**

   - Sign in to your Kaggle account.
   - Navigate to "Kernels" and create a new notebook.
   - Upload the downloaded notebook file.

3. **Run the Notebook:**

## Steps Taken

1. **Determining the Training Platform**

   - **Google Colab**: Efficient GPU but limited in access time.
   - **AzureML Compute Instances and Clusters**: Offers $100 of free access to different CPUs for students.
     However, using a custom virtual environment with Python==3.10 resulted in the notebook reverting to Python 3.19.4,
     which is incompatible with the NeMo toolkit models requiring Python>=3.10.
   - **Kaggle**: Given the limited time (30 hours per user \* 3 team members) and total control of the environment,
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

This research provides a comparative analysis of various ASR models on an Egyptian dialect dataset, examining how model architecture and configurations impact performance, specifically in terms of Word Error Rate (WER). Our experiments yielded the following insights and answers to key research questions:

### Key Findings and Recommendations

1. **Impact of Model Dimensions on WER**  
   Models with larger encoder dimensions (512 or higher) demonstrated worse performance and often produced empty predictions for substantial portions of the test data. This issue persisted regardless of the number of epochs, suggesting that large dimensions may lead to overfitting or ineffective learning when trained on datasets of limited size or diversity.

   - **Recommendation**: Small (176) or medium (256) encoder dimensions are optimal for achieving a balance between accuracy and computational efficiency, especially for dialectal ASR tasks.

2. **Effect of Training Epochs on Model Convergence**  
   Despite extensive training (up to 100 epochs for some models), both the Conformer-Hybrid and Transducer models failed to converge. This may be due to a mismatch between these models' complexity and the size of the dataset, which limits their ability to generalize effectively.

   - **Recommendation**: For datasets of similar size, simpler architectures like CTC or Jasper, which achieved WERs of 13.2% and 24.4% respectively, are recommended. Complex architectures may require more data or additional regularization techniques to reach convergence.

3. **Benefits of SpecAugment for Handling Noise**  
   Applying SpecAugment during training improved model robustness to noisy audio tracks, as observed in models that incorporated this augmentation. By masking portions of the spectrogram, SpecAugment effectively helps the model generalize to varied audio conditions, which is essential for real-world ASR applications with background noise.

   - **Recommendation**: Incorporate SpecAugment for ASR models aimed at dialectal or noisy datasets to enhance robustness and improve WER.

4. **Best Performing Architecture**  
   The CTC-based Conformer model, with a character label type and an encoder dimension of 176, achieved the best WER of 13.2% after 40 epochs. This result indicates that for limited-data scenarios, simpler architectures may outperform hybrid or transducer-based models, particularly when they are optimized with data augmentation and medium-sized encoders.
   - **Conclusion**: The CTC model’s simplicity, coupled with an appropriate encoder size and SpecAugment, makes it the most suitable choice for Egyptian dialect ASR.

### Future Work and Implications

For future ASR research on dialectal datasets, the findings suggest focusing on small to medium models with effective data augmentation techniques. Investigating other regularization methods or advanced training strategies, such as curriculum learning or semi-supervised training, may further improve convergence in complex architectures. Additionally, increasing the dataset size could enable larger models to fully utilize their capacity, potentially closing the gap between simple and complex model performance.

In conclusion, by identifying the optimal balance between model complexity and dataset constraints, this study provides actionable insights into developing efficient and robust ASR models for dialectal variations.
