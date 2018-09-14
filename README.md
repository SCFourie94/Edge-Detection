# Edge-Detection
Using LoG as the first method. 
Second method is to use Gaussian first and then Laplacian.

Mask sizes of 7, 13, and 25 are used with sigma values 1, 2, 4 respectively.

Formulas are used to generate the respective masks.

After the images has been convolved with the masks, the zero crossing is generated.
