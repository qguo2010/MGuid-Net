# MGuid-Net
Code repository for our paper "Multi-Guidance CNNs for Salient Object Detection". [Paper]() is available.

# Usage
1. Environment configurations
    * pytorch 1.8.1
    * python 3.6.13
2. Train/Test
    * Train
        * Download the source codes of MGuid-Net, then
        ```python
        Run 'train_MGuid.py'
        ```
        * In our experiments, the batch size is set to 7, and the image size is 352 $\times$ 352. If your hardware is enough, you can increase the batch size and image size to achieve better experimental results.
    * Test
      
        * Download other testing datasets ([DUT-OMRON](http://saliencydetection.net/dut-omron/), [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html), [HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency.html), [PASCAL-S](http://www.cbi.gatech.edu/salobj/)) except DUTS-TE provided by us.
        * Set your dataset path, then
        ```python
        Run 'test_MGuid.py'
        ```
3. To facilitate comparison, the pre-computed saliency maps can be obtained from [baidu yun]() (code:).

