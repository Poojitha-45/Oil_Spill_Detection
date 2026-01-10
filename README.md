🌊 Oil Spill Detection Using SAR Satellite Imagery
   An End-to-End Deep Learning System for Environmental Monitoring


             --📌 Abstract--

Oil spills are among the most severe environmental disasters, causing long-term damage to marine ecosystems, coastal regions, and human livelihoods. Traditional monitoring methods are time-consuming and limited in coverage.

This project presents an AI-driven Oil Spill Detection and Severity Assessment System using Synthetic Aperture Radar (SAR) satellite imagery and a U-Net deep learning architecture. The system performs pixel-level segmentation to accurately identify oil spill regions, assesses spill severity, and provides an interactive web-based user interface for real-time analysis.


         --🧭 Table of Contents--

1.Introduction
2.Problem Statement
3.Why SAR Images?
4.Dataset Description
5.Project Workflow
6.Exploratory Data Analysis (EDA)
7.Data Preprocessing
8.Model Architecture – U-Net
9.Model Training
10.Evaluation Metrics
11.Error Analysis & Improvements
12.Post-Processing
13.Severity Assessment Logic
14.Web Application (UI)
15.Deployment
16.Results & Sample Outputs
17.Limitations
18.Future Enhancements
19.Conclusion


            ---1️⃣ Introduction---

Marine oil spills are difficult to monitor due to their dynamic nature and large spatial extent. Satellite imagery provides a scalable solution, and SAR images are especially effective as they operate independently of weather and lighting conditions.

This project builds a complete AI pipeline, from raw satellite data to a deployed web application, enabling users to upload SAR images and instantly receive oil spill detection results


          ---2️⃣ Problem Statement---

The goal of this project is to:

--->Detect oil spills from SAR satellite images
--->Segment oil and non-oil regions at pixel level
--->Quantify the severity of the spill
--->Provide an easy-to-use interface for non-technical users
--->Deploy the solution as a publicly accessible application


         ---3️⃣ Why SAR Images?---

Unlike optical images, SAR images:

-->Work day and night
-->Penetrate clouds and fog
-->Highlight oil spills as bright (white) regions
-->Are widely used in real-world maritime surveillance
Important:
In this project:
-->White pixels → Oil Spill
-->Black pixels → Water


           ---4️⃣ Dataset Description---

Image Type: Grayscale SAR images
Labels: Binary segmentation masks

Mask Encoding:
White (1) → Oil
Black (0) → Water




           ---5️⃣ Project Workflow---


Data Collection
      ↓
EDA
      ↓
Preprocessing
      ↓
U-Net Model Training
      ↓
Evaluation
      ↓
Post-Processing
      ↓
Severity Analysis
      ↓
Web App Deployment


                 ---6️⃣ Exploratory Data Analysis (EDA)---


EDA was performed to understand:

Pixel intensity distribution
Image dimensions
Mask uniqueness
Oil vs water pixel ratio

Key observations:

SAR images are single-channel (grayscale)
Oil regions appear brighter
Class imbalance exists (water dominates)

<img width="930" height="674" alt="image" src="https://github.com/user-attachments/assets/b065cc83-bd92-4844-9ab1-d240d469e918" />




                       ---7️⃣ Data Preprocessing---


Steps applied:

Grayscale normalization (0–1)
Resizing to 256×256
Binary thresholding for masks
Optional data augmentation:
Horizontal flip
Vertical flip
Rotation


                      ---8️⃣ Model Architecture – U-Net---

The U-Net architecture is chosen due to its effectiveness in segmentation tasks.

Architecture Highlights:

-->Encoder (contracting path) extracts contextual features
-->Decoder (expanding path) restores spatial resolution
-->Skip connections preserve fine-grained details
-->Final sigmoid layer outputs pixel-wise probabilities



                               ----9️⃣ Model Training---

Loss Function:

Binary Cross-Entropy + Dice Loss
Optimizer: Adam
Learning Rate: 0.001
Epochs: 15+
Batch Size: 8
Training was checkpointed to allow resume after interruption.

📷 Add training & validation loss curves here

                         ---🔁 Training Concepts Explained---

Forward Pass: Image → Encoder → Decoder → Mask

Loss Calculation: Measures segmentation error

Backpropagation: Updates weights

Epoch: One full pass through dataset

Batch: Subset of images per iteration

Validation: Measures generalization


                         ---🔍 10️⃣ Evaluation Metrics---

Accuracy is misleading for segmentation, so we used:
Dice Coefficient
Intersection over Union (IoU)
Precision & Recall
Pixel-level Confusion Matrix

📷 Add confusion matrix image here



                    ----11️⃣ Error Analysis & Improvements---

Observed issues:

Boundary inaccuracies
Small false positives

Applied improvements:
Lowered threshold for SAR contrast
Morphological post-processing
Data augmentation
Longer training


                                 ----12️⃣ Post-Processing-----

To reduce noise:

Morphological opening
Morphological closing
This removes isolated false detections and smooths boundaries.

📷 Add before & after post-processing comparison



                             -----13️⃣ Severity Assessment Logic---

Severity is calculated based on oil pixel percentage:

       Oil  Coverage	                        Severity	                               Risk
            0%	                                 None	                             threat
          < 5%	                                 Low	                             Minimal impact
          5–20%	                                Medium	                             Moderate risk
          > 20%	                                 High	                              Severe threat



                                  -----14️⃣ Web Application (UI)----

A Streamlit-based web application was developed with:

SAR image upload
Run Analysis button
Probability map
Binary mask
Overlay visualization
Severity analysis
PDF report download

📷 Add UI screenshot here


                            ----15️⃣ Deployment----

The application is deployed using Streamlit Community Cloud.

-->Public access
-->CPU-based inference
-->GitHub-linked deployment

🔗 Live App Link:
(Add your deployed URL here)


                            ---16️⃣ Results & Sample Outputs---

📷 Add multiple result samples here

Input SAR image

Probability map

Binary mask

Overlay image

Severity output


                             ----17️⃣ Limitations----

Works only on SAR images
Not trained on optical imagery
Performance depends on SAR quality


                                 ----18️⃣ Future Enhancements---

Multi-class spill categorization
Temporal spill tracking
Location-based alerts
Integration with GIS systems
Mobile-friendly UI
Real-time satellite feed


                                      ----19️⃣ Conclusion----

This project demonstrates a complete AI solution, from raw satellite data to a deployed application, capable of detecting and assessing oil spills with high precision. It highlights the practical use of deep learning in environmental monitoring and disaster management.



---📜 Acknowledgements---

Mentor guidance
Open SAR datasets
