A multitask model for building height Estimation and shadow, footprint mask generation, trained on the unbalanced GRSS-2023 dataset.
Domain adapted for India with Google Open Building data, Introduced a novel seam carving based augmentation technique. 
Trained on UNet architecture with ASSP based encoder and two decoder one for height estimation known as regression deoder and other for shadow mask and footprint generation known as segmentation decoder, 
regression decoder uses windowed cross attention to query about shadow and footprint information from the segmentation decoder.![Delhi_Estimates](https://github.com/user-attachments/assets/82579228-b7dc-499e-9941-b6ddcae288a1)
