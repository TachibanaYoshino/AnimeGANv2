# AnimeGANv2   

[Open Source]. The improved version of AnimeGAN.  
-----  
**Focus:**  
|-Anime style-|-film-|-Picture amount-|-Quality-|-Download link-|
| ------ | ------ |------ |------ |------ |
|-Miyazaki Hayao-|-THE Wind Rises-|-1752-|-1080p-|- ***TBD*** -|
|-Makoto Shinkai-|-Weathering with you-|-5908-|-BD-|-***TBD***-|
|-Kon Satoshi-|-Paprika-|-1255-|-BD-|-***TBD***-|
  
  
**News:**    
```yaml
The improvement directions of AnimeGAN+ mainly include the following 4 points:  
```
&ensp;&ensp; 1. Solve the problem of high-frequency artifacts in the generated image.  
&ensp;&ensp; 2. It is easy to train and directly achieve the effects in the paper.  
&ensp;&ensp; 3. Further reduce the number of parameters of the generator network.**(8.17 Mb)**  
&ensp;&ensp; 4. Use new high-quality style data, which come from BD movies as much as possible.  
   
   &ensp;&ensp;&ensp;&ensp;&ensp;  AnimeGAN A can be accessed from [here](https://github.com/TachibanaYoshino/AnimeGAN).    
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
  > `python test.py --checkpoint_dir  checkpoint/generator_Hayao_weight  --test_dir dataset/test/real --style_name H`  
  
### 2. Convert video to anime   
  > `python video2anime.py  --video video/input/お花見.mp4  --checkpoint_dir  ../checkpoint/generator_Paprika_weight`  
    
____  
## Results  
   
:heart_eyes:  Photo  to  Paprika  Style  
  
![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo/1%20(37).jpg) ![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo_result/1%20(37).jpg)  
![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo/1%20(1).jpg) ![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo_result/1%20(1).jpg)  
![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo/1%20(31).jpg) ![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo_result/1%20(31).jpg)  
![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo/1%20(21).jpg) ![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo_result/1%20(21).jpg)  
![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo/1%20(60).jpg) ![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo_result/1%20(60).jpg)  
![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo/1%20(23).jpg) ![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo_result/1%20(23).jpg)  
![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo/1%20(24).jpg) ![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo_result/1%20(24).jpg)  
![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo/1%20(46).jpg) ![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo_result/1%20(46).jpg)  
![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo/1%20(30).jpg) ![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo_result/1%20(30).jpg)  
![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo/1%20(28).jpg) ![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo_result/1%20(28).jpg)  
![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo/1%20(44).jpg) ![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo_result/1%20(44).jpg)  
____  
:heart_eyes:  Photo  to  Hayao  Style   

____  
:heart_eyes:  Photo  to  Shinkai  Style   
**TBD**
