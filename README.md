# nc-2t2r



## Requirements
* Follow the official guide to install [Pytorch](https://pytorch.org/get-started/locally/)
* Install requirements: `pip install -r requirements.txt`

## Training and Evaluation
* Train neural network with 2T2R model:
  ```bash
  python train.py --quant --diff --save m_2t2r_q
  ```

* Train neural network with 1T1R model:
  ```bash
  python train.py --quant --save m_1t1r_q
  ```

* Train ideal neural network:
  ```bash
  python train.py --save m_ideal
  ```

* Run evaluation with specific checkpoint path:
  ```bash
  python eval.py checkpoints/m_2t2r_q  # or m_1t1r_q / m_ideal
  ```
