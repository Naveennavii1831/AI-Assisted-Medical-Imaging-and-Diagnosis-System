\# Model Evaluation Report



\## AI-Assisted Medical Imaging and Diagnosis System



\### 1. Project Overview



This project explores the use of deep learning for multilabel classification of chest X-ray images. It uses DenseNet121 to generate predictions for 14 labels in the ChestMNIST dataset.



The system also includes a prototype interface for image upload, prediction display, and Grad-CAM visualization.



\*\*Intended use:\*\* Educational and research prototyping only. The system has not been clinically validated and must not be used to diagnose or treat patients.



\### 2. Dataset



\* Dataset: ChestMNIST

\* Dataset source: MedMNIST

\* Evaluation split: Official test split

\* Number of test images: 22,433

\* Number of output labels: 14

\* Task: Multilabel image classification



Each image can have multiple positive labels.



\### 3. Model Architecture



Both experimental models use DenseNet121 with a modified final classification layer that produces 14 outputs.



The models were trained using different approaches:



\* \*\*V2:\*\* Binary cross-entropy training objective without the class-weighting modification used in V3.

\* \*\*V3:\*\* Class-weighted binary cross-entropy to address label imbalance.



The model outputs are converted to probabilities using the sigmoid function. A threshold determines whether each label is predicted as positive.



\### 4. Image Preprocessing



The evaluation pipeline uses the following transformations:



1\. Resize images to 64 × 64 pixels.

2\. Convert grayscale images to three channels.

3\. Convert images to tensors.

4\. Normalize using the ImageNet mean and standard deviation.



The preprocessing configuration matches the existing training configuration.



\### 5. Test-Set Evaluation Results



Both models were evaluated on the official ChestMNIST test split.



| Metric                    |     V2 |     V3 |

| ------------------------- | -----: | -----: |

| Decision threshold        |   0.15 |   0.60 |

| Element accuracy          | 88.41% | 84.87% |

| Micro precision           | 17.31% | 11.97% |

| Micro recall              | 31.86% | 29.55% |

| Micro F1-score            | 22.43% | 17.04% |

| Macro precision           |  8.76% |  8.41% |

| Macro recall              | 14.17% | 20.47% |

| Macro F1-score            |  8.52% | 10.78% |

| Macro ROC-AUC             | 0.5891 | 0.5955 |

| Binary cross-entropy      | 0.1931 | 0.4977 |

| Predicted positive labels | 30,395 | 40,755 |



The values are rounded for readability. The exact metrics are available in `docs/final\_model\_test\_comparison.json`.



\### 6. Interpretation



V2 achieved higher micro precision, micro recall, micro F1-score, element accuracy, and lower binary cross-entropy than V3.



V3 achieved higher macro F1-score, macro recall, and macro ROC-AUC.



Micro metrics aggregate predictions across all labels. Macro metrics calculate each label's metric separately and then average across labels, giving each label equal weight.



Element accuracy should be interpreted cautiously because label imbalance can make this metric appear high even when disease detection performance is limited.



The ROC-AUC values indicate that the models have limited ability to distinguish positive from negative labels.



\### 7. Model Selection



V2 is retained as the primary experimental model because it achieved the higher micro F1-score and better overall balance between micro precision and micro recall at the selected threshold.



V3 is retained as an experimental comparison because it achieved a higher macro F1-score.



The thresholds were selected using validation data. The official test split was subsequently used to evaluate the models at those thresholds.



The model-selection decision is specific to these experiments and does not establish clinical suitability.



\### 8. Limitations



1\. The measured performance is limited and is not sufficient to support clinical diagnosis.

2\. The dataset contains imbalanced labels.

3\. Results depend on the selected decision thresholds.

4\. Validation and test performance may not represent performance on images from other hospitals, scanners, or patient populations.

5\. The current classification models have not been clinically validated.

6\. Grad-CAM visualizations indicate image regions influencing model outputs but do not prove that the model's reasoning is medically correct.

7\. Predictions on CT or MRI images are not validated by this ChestMNIST-trained classification model.

8\. The project is an educational prototype, not a medical device.



\### 9. Reproducibility and Project Artifacts



Important project files include:



\* `src/train\_model\_v2.py`

\* `src/train\_model\_v3.py`

\* `src/evaluate\_model\_v2.py`

\* `src/evaluate\_final\_models.py`

\* `src/compare\_models\_v2\_v3.py`

\* `docs/training\_history\_v2.json`

\* `docs/training\_history\_v3.json`

\* `docs/threshold\_tuning\_v2.json`

\* `docs/threshold\_tuning\_v3.json`

\* `docs/model\_comparison\_v2\_v3.json`

\* `docs/final\_model\_test\_comparison.json`



\### 10. Conclusion



The project demonstrates an experimental deep-learning pipeline for multilabel chest X-ray classification, model evaluation, image upload, and visualization.



V2 is the current primary experimental model based on its micro F1-score and related metrics. Further work is needed to improve classification performance, evaluate individual labels, strengthen reproducibility, and complete the remaining application workflow.



\*\*Final notice: For educational and research purposes only. Not for clinical use.\*\*
