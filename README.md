# Multi-Scale Fully Convolutional Network

In this repository, we design two branches with convolutional layers in different kernel sizes in each layer of the encoder to capture multi-scale features. Besides, a channel attention block and a global pooling module are utilized to enhance channel consistency and global contextual consistency. Substantial experiments are conducted on both 2D RGB images datasets and 3D spatial-temporal datasets.

The detailed results can be seen in the [Land Cover Classification from Remote Sensing Images Based on Multi-Scale Fully Convolutional Network](https://arxiv.org/ftp/arxiv/papers/2009/2009.02130.pdf).

The related repositories include:
* [MACU-Net](https://github.com/lironui/MACU-Net)->A revised U-Net structure.
* [MAResU-Net](https://github.com/lironui/MAResU-Net)->Another type of attention mechanism with linear complexity.

If our code is helpful to you, please cite:

`Li, R., Zheng, S., Duan, C. *, Wang, L., & Zhang, C. (2021). Land Cover Classification from Remote Sensing Images Based on Multi-Scale Fully Convolutional Network. Geo-spatial Information Science.`

Acknowlegement:
------- 
Thanks to the providers of the folloing open-source datasets:
[WHDLD](https://sites.google.com/view/zhouwx/dataset?authuser=0#h.p_ebsAS1Bikmkd)
[GID](https://x-ytong.github.io/project/GID.html)
[2015&2017](http://gpcv.whu.edu.cn/data/3DFGC_pages.html)

Requirements：
------- 
```
numpy >= 1.16.5
PyTorch >= 1.3.1
sklearn >= 0.20.4
tqdm >= 4.46.1
imageio >= 2.8.0
```

Network:
------- 
![network](https://github.com/lironui/MSFCN/blob/master/Fig/network.png)  
Fig. 1.  The structure of the proposed Multi-Scale Fully Convolutional Network.

Result:
------- 
![Result1](https://github.com/lironui/MSFCN/blob/master/Fig/2D_zoom.png)  
Fig. 2. Visualization of results on the WHDLD and GID datasets.

![Result2](https://github.com/lironui/MSFCN/blob/master/Fig/3D_zoom.png)  
Fig. 3. Visualization of results on the 2015 and 2017 datasets.
