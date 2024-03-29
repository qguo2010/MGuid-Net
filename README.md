# MGuid-Net
S. Hui, Q. Guo, X. Geng, and C. Zhang, [Multi-guidance CNNs for salient object detection](https://dl.acm.org/doi/10.1145/3570507), *ACM Transactions on Multimedia Computing Communications and Applications*, vol. 19, no. 3, article 117, Feb. 2023.

# Framework
![architecture](./framwork.png)


# Usage
1. Environment configurations
    * pytorch 1.8.1
    * python 3.6.13
2. Train/Test
    * Train
        * Download the source codes of MGuid-Net and the training datasets ([DUTS-TR](http://saliencydetection.net/duts/)), then
        ```python
        Run 'train_MGuid.py'
        ```
        * In our experiments, the batch size is set to 7, and the image size is 352 $\times$ 352. If your hardware is enough, you can increase the batch size and image size to achieve better experimental results.
    * Test
      
        * Download the testing datasets ([DUTS-TE](http://saliencydetection.net/duts/), [DUT-OMRON](http://saliencydetection.net/dut-omron/), [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html), [HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency.html), [PASCAL-S](http://www.cbi.gatech.edu/salobj/)).
        * Set your dataset path, then
        ```python
        Run 'test_MGuid.py'
        ```
3. To facilitate comparison, the pre-computed saliency maps can be obtained from [baidu yun](https://pan.baidu.com/s/16oEoBJ1Jc-gsFox2WeSC-w) (access code: k8bh).

