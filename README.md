# AnimeGANv2   

[Open Source]. The improved version of AnimeGAN.  
Landscape photos/videos to anime  
-----  
**Focus:**  
<table border="1px ridge">
	<tr align="center">
	    <th>Anime style</th>
	    <th>Film</th>  
	    <th>Picture Number</th>  
      <th>Quality</th>
      <th>Download Style Dataset</th>
	</tr >
	<tr align="center">
      <td>Miyazaki Hayao</td>
      <td>The Wind Rises</td>
      <td>1752</td>
      <td>1080p</td>
	    <td rowspan="3"><a href="https://github.com/TachibanaYoshino/AnimeGANv2/releases/tag/1.0">Link</a></td>
	</tr>
	<tr align="center">
	    <td>Makoto Shinkai</td>  
	    <td>Your Name & Weathering with you</td>
      <td>1445</td>
      <td>BD</td>
	</tr>
	<tr align="center">
	    <td>Kon Satoshi</td>
	    <td>Paprika</td>
      <td>1284</td>
      <td>BDRip</td>
	</tr>
</table>  
  
  
  &ensp;&ensp;&ensp;&ensp;&ensp;Different styles of training have different loss weights!
  
**News:**    
```yaml
The improvement directions of AnimeGANv2 mainly include the following 4 points:  
```  
- [x] 1. Solve the problem of high-frequency artifacts in the generated image.  
- [x] 2. It is easy to train and directly achieve the effects in the paper.  
- [x] 3. Further reduce the number of parameters of the generator network. **(generator size: 8.17 Mb)**, The lite version has a smaller generator model.  
- [x] 4. Use new high-quality style data, which come from BD movies as much as possible.  
   
   &ensp;&ensp;&ensp;&ensp;&ensp;  AnimeGAN can be accessed from [here](https://github.com/TachibanaYoshino/AnimeGAN).  
   
**(suitable for Google Colab) the out-of-the-box Jupyter Notebook link is [here](https://colab.research.google.com/github/CyFeng16/MVIMP/blob/master/docs/MVIMP_AnimeGANv2_Demo.ipynb). **
___  

## Requirements  
- python 3.6  
- tensorflow-gpu 
   - tensorflow-gpu 1.8.0  (ubuntu, GPU 1080Ti or Titan xp, cuda 9.0, cudnn 7.1.3)  
   - tensorflow-gpu 1.15.0 (ubuntu, GPU 2080Ti, cuda 10.0.130, cudnn 7.6.0)  
- opencv  
- tqdm  
- numpy  
- glob  
- argparse  
  
## Usage  

### 1. Download vgg19    
  > [vgg19.npy](https://github.com/TachibanaYoshino/AnimeGAN/releases/tag/vgg16%2F19.npy)  

### 2. Download Train/Val Photo dataset  
  > [Link](https://github.com/TachibanaYoshino/AnimeGAN/releases/tag/dataset-1)  

### 3. Do edge_smooth  
  > `python edge_smooth.py --dataset Hayao --img_size 256`  
  
### 4. Calculate the three-channel(BGR) color difference  
  >  `python data_mean.py --dataset Hayao`  
  
### 5. Train  
  >  `python main.py --phase train --dataset Hayao --data_mean [13.1360,-8.6698,-4.4661] --epoch 101 --init_epoch 10`  
  >  For light version: `python main.py --phase train --dataset Hayao --data_mean [13.1360,-8.6698,-4.4661]  --light --epoch 101 --init_epoch 10`  
  
### 6. Extract the weights of the generator  
  >  `python get_generator_ckpt.py --checkpoint_dir  ../checkpoint/AnimeGAN_Hayao_lsgan_300_300_1_2_10_1  --style_name Hayao`  

### 7. Inference      
  > `python test.py --checkpoint_dir  checkpoint/generator_Hayao_weight  --test_dir dataset/test/HR_photo --style_name Hayao/HR_photo`  
  
### 8. Convert video to anime   
  > `python video2anime.py  --video video/input/お花見.mp4  --checkpoint_dir  checkpoint/generator_Paprika_weight`  
    
____  
## Results  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/AnimeGANv2.png)   
     
____ 
:heart_eyes:  Photo  to  Paprika  Style  
  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/37.jpg)   
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/38.jpg)     
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/6.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/7.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/9.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/21.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/44.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/1.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/8.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/11.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/5.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/15.jpg)   
____  
:heart_eyes:  Photo  to  Hayao  Style   
  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/34.jpg)   
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/10.jpg)     
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/15.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/35.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/39.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/42.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/44.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/41.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/32.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/11.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/5.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/18.jpg)    
____  
:heart_eyes:  Photo  to  Shinkai  Style   
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Shinkai/concat/7.jpg)   
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Shinkai/concat/9.jpg)     
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Shinkai/concat/11.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Shinkai/concat/15.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Shinkai/concat/17.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Shinkai/concat/22.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Shinkai/concat/27.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Shinkai/concat/33.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Shinkai/concat/32.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Shinkai/concat/21.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Shinkai/concat/3.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Shinkai/concat/26.jpg)  
  
## License  
This repo is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications. Permission is granted to use the AnimeGANv2 given that you agree to my license terms.
## Author  
Xin Chen
