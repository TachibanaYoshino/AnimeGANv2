# AnimeGANv2   

[Open Source]. The improved version of AnimeGAN.  
-----  
**Focus:**  
|Anime style|Film|Picture Number|Quality|Download link|
| :------: | :------: |:------: | :------: | :------: |
|Miyazaki Hayao|The Wind Rises|1752|1080p| ***TBD*** |
|Makoto Shinkai|Your Name|1642|BDRip|***TBD***|
|Kon Satoshi|Paprika|1255|BDRip|***TBD***|
  
  &ensp;&ensp;&ensp;&ensp;&ensp;Different styles of training have different loss weights!
  
**News:**    
```yaml
The improvement directions of AnimeGANv2 mainly include the following 4 points:  
```  
- [x] 1. Solve the problem of high-frequency artifacts in the generated image.  
- [x] 2. It is easy to train and directly achieve the effects in the paper.  
- [x] 3. Further reduce the number of parameters of the generator network. **(generator size: 8.17 Mb)**  
- [x] 4. Use new high-quality style data, which come from BD movies as much as possible.  
   
   &ensp;&ensp;&ensp;&ensp;&ensp;  AnimeGAN can be accessed from [here](https://github.com/TachibanaYoshino/AnimeGAN).  
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

### 1. Inference      
  > `python test.py --checkpoint_dir  checkpoint/generator_Hayao_weight  --test_dir dataset/test/HR_photo --style_name Paprika/HR_photo`  
  
### 2. Convert video to anime   
  > `python video2anime.py  --video video/input/お花見.mp4  --checkpoint_dir  checkpoint/generator_Paprika_weight`  
    
____  
## Results  
   
:heart_eyes:  Photo  to  Paprika  Style  
  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/34.png)   
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/10.png)     
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/15.png)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/35.png)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/39.png)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/42.png)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/44.png)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/41.png)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/32.png)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/11.png)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/5.png)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/18.png)   
____  
:heart_eyes:  Photo  to  Hayao  Style   
  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/34.png)   
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/10.png)     
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/15.png)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/35.png)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/39.png)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/42.png)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/44.png)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/41.png)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/32.png)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/11.png)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/5.png)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/18.png)    
____  
:heart_eyes:  Photo  to  Shinkai  Style   
**TBD**
