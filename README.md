 Oil Spill Detection Using SAR Satellite Imagery
 
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



<img width="416" height="727" alt="image" src="https://github.com/user-attachments/assets/1069f199-994f-49c0-a8d5-60240233b6a4" />



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


👉Tech Stack:

Python

PyTorch

OpenCV

NumPy

Streamlit

SAR Satellite Imagery

Git & GitHub


                               ----9️⃣ Model Training---

Loss Function:

Binary Cross-Entropy + Dice Loss

Optimizer: Adam

Learning Rate: 0.001

Epochs: 15+

Batch Size: 8

Training was checkpointed to allow resume after interruption.


<img width="587" height="352" alt="image" src="https://github.com/user-attachments/assets/2c32dfc4-20a7-4d68-a44e-8918279f6339" />

                   
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

<img width="1450" height="805" alt="image" src="https://github.com/user-attachments/assets/f32b3d49-89a0-47ab-a549-0eea6777d312" />




<img width="819" height="691" alt="image" src="https://github.com/user-attachments/assets/5056dd3b-b4dd-464f-bc0f-fe756c4070d1" />



                    ----11️⃣ Error Analysis & Improvements---

Observed issues:

Boundary inaccuracies

Small false positives


Applied improvements:

Lowered threshold for SAR contrast

Morphological post-processing

Data augmentation

Longer training



<img width="1186" height="868" alt="image" src="https://github.com/user-attachments/assets/441c9ab2-369d-4997-b7fb-3c8f99118739" />



                                 ----12️⃣ Post-Processing-----

To reduce noise:

Morphological opening

Morphological closing

This removes isolated false detections and smooths boundaries.

<img width="1264" height="824" alt="image" src="https://github.com/user-attachments/assets/53392942-0281-468e-9597-dfc1e2d4fcb2" />




                             -----13️⃣ Severity Assessment Logic---

Severity is calculated based on oil pixel percentage:

       Oil  Coverage	                        Severity	                               Risk
            0%	                                 None	                             threat
          < 5%	                                 Low	                             Minimal impact
          5–20%	                              Medium	                           Moderate risk
          > 20%	                               High	                              Severe threat



                                 
                                  
                                  -----14️⃣ Web Application (UI)----

A Streamlit-based web application was developed with:

SAR image upload

Run Analysis button

Probability map

Binary mask

Overlay visualization

Severity analysis

PDF report download


<img width="1903" height="771" alt="image" src="https://github.com/user-attachments/assets/668b166c-8d0c-4705-b1a4-7b2fbaa95de3" />



                            ----15️⃣ Deployment----

The application is deployed using Streamlit Community Cloud.

-->Public access

-->CPU-based inference

-->GitHub-linked deployment

🔗 Live Application Demo

You can access the fully deployed and interactive version of this project here:


https://oil-spilldetection-2k25.streamlit.app/


                            ---16️⃣ Results & Sample Outputs---


Input SAR image

<img width="256" height="256" alt="palsar_20" src="https://github.com/user-attachments/assets/4db132f0-28b1-4364-900b-093b5eed9a58" />




Probability map

<img width="480" height="497" alt="image" src="https://github.com/user-attachments/assets/782a00fa-4b69-429d-aaec-b6753ada82d5" />




Binary mask

<img width="956" height="595" alt="image" src="https://github.com/user-attachments/assets/2479c8ef-cdbc-4c95-93c3-9ebf32fc2d59" />



Overlay image

<img width="417" height="500" alt="image" src="https://github.com/user-attachments/assets/f3bed413-e758-47c4-af5c-5079a8ba9ab7" />




Severity output

<img width="542" height="482" alt="image" src="https://github.com/user-attachments/assets/3ff9e62f-922a-48eb-9a89-dd100cb0586c" />



Overall UI and output of the model with the downloaded report

<img width="1864" height="794" alt="image" src="https://github.com/user-attachments/assets/02ac85d1-6b4d-4a0d-aa2c-b8f01ad5c503" />
<img width="1798" height="773" alt="image" src="https://github.com/user-attachments/assets/ff57efe4-70ef-4e06-8e2c-eea0aa1b9180" />
<img width="562" height="830" alt="image" src="https://github.com/user-attachments/assets/8c330de3-bc82-49f1-8315-8e5770314eb5" />
<img width="621" height="801" alt="image" src="https://github.com/user-attachments/assets/30b0b34d-a467-4080-b588-4aa7f468aeaa" />






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


👩‍💻 Author

Name: Poojitha

Role: AI / ML Developer

Project Type: Academic / Research / Internship Project




---📜 Acknowledgements---

Mentor guidance

Open SAR datasets
