# Phoenix_Transformation
Hi, I'm interested in the synthesis of realistic biological systems, and this repository is dedicated to the training, synthesis, and application of realistic transcriptomes.
This respository heavily borrows, i.e., cloning with a tiny bit of modification, from StyleGAN2-ADA and SRGAN.
Due to file size limitations, only a small portion of images are included in the training library for SRGAN, StyleGAN-ADA, and pix2pix.
The full training library and saved model for StyleGAN2-ADA can be accessed from my google drive at:
https://drive.google.com/drive/folders/1grCMEqRQ0qSmp2a7z9PICSL6V_6fdm09?usp=sharing

The full training library for SRGABN can be accessed at:
https://drive.google.com/drive/folders/1Qz5xjVuTKpN9jl8qPWVvjX_flx0xOs63?usp=sharing

SRGAN is cloned from: https://github.com/Lornatang/SRGAN-PyTorch.

StyleGAN2-ADA is cloned from: https://github.com/dvschultz/stylegan2-ada-pytorch.

Main functionalities of this repository are:
1, Transcriptome_image_rendering.R in the Rs folder can be used for transcriptome image production from a gene expression table.
2, CNN_2layer_binary_adam.py from the CNNs folder can be used for transcriptome image classification model training.
3, XGBoost_classification.py from the XGBoost folder can be used for transcriptome classification with the XGBoost classifier.
4, train.py from the StyleGAN folder can be used to train models for transcriptome images.
5, generate.py from the StyleGAN folder can be used to synthesize transcriptome images from trained models with class-specific seeds.
6, train512_1024.py from the SRGAN folder can be used to train transcriptome images for super-resolution from 512x512 to 1024x1024 pixels.
7, train768_3072.py from the SRGAN folder can be used to train transcriptome images for super-resolution from 768x768 to 3072x3072 pixels.
8, Test_image_folder_2fold.py from the SRGAN folder can be used to super-resolution folders of 512x512 transcriptome image to 1024x1024 images.
9, Test_image_folder_4fold.py from the SRGAN folder can be used to super-resolution folders of 768x768 transcriptome image to 3072x3072 images.
10, Image_quantification_for_folder_G57_1.py from the SRGAN folder can be used to convert transcritome images of 3072x3072 resolution to gene expression lists.
The trained models for transcriptome image super resolution can be found in the SRGAN/epochs folder.
