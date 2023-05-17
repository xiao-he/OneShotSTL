# OneShotSTL (VLDB 2023)

## This is an official implementation of OneShotSTL

Xiao He, Ye Li, Jian Tan, Bin Wu, Feifei Li. "OneShotSTL: One-Shot Seasonal-Trend Decomposition For Online Time Series Anomaly Detection And Forecasting" in Proceedings of the VLDB Endowment 16, 06 (2023), 1399-1412. [Paper](https://www.vldb.org/pvldb/vol16/p1399-he.pdf)

with a Java artifact (java/OneShotSTL/OneShotSTL.jar) and Python scripts/notebooks for reproducing experiments in the paper.

## Key results
:star2: OneShotSTL is an online/incremental seasonal-trend decomposition method with **O(1) update complexity**, which can be used for online time series anomaly detection and forecasting. 

:star2: It takes around **20 μs** for OneShotSTL to process each data point on a typical commodity laptop using a single CPU core.

:star2: On univariate long-term time series forecasting tasks, OneShotSTL is more than **1000 times faster** than the state-of- the-art deep learning/Transformer based methods with **comparable or even better accuracy**.

![alt text](https://github.com/xiao-he/OneShotSTL/blob/main/pic/table5.png)


## Get Started
1. Install Python=3.8 and Java
2. Install Python requirements
```
pip install -r requirements.txt
```
3. Download TSB-UAD-Public.zip from [TSB-UAD](https://github.com/TheDatumOrg/TSB-UAD) and unzip it into data folder
4. Download all_six_datasets.zip from [Autoformer](https://github.com/thuml/Autoformer) and unzip it into data folder
5. Preprocess TSB-UAD datasets: 
```
python preprocess_TSB-UAD.py 
```
6. Preprocess forecast datasets: 
```
python preprocess_forecast.py 
```

## Reproduce
1. Synthetic experiments in Figure 5: 
```
exp_synthetic.ipynb
```
2. Scalability experiments in Figure 7: 
```
exp_scalability.ipynb
```
3. Univariate Time Series Anomaly detection experiments in Table 3:
```
exp_TSB-UAD.ipynb
```
4. Univariate Time Series Forecast experiments in Table 5:
```
exp_forecast.ipynb
```

## Compile Java maven project
If you would like to modify the java code, you can recompile the maven project as following (maven 3.6.3 is needed):
```
cd java/OneShotSTL
mvn clean compile assembly:single
mv target/OneShotSTL-1.0-SNAPSHOT-jar-with-dependencies.jar OneShotSTL.jar
```

## Citation
If you find this repo useful, please cite our paper.
```
@article{he2023oneshotstl,
	title={{OneShotSTL: One-Shot Seasonal-Trend Decomposition For Online Time Series Anomaly Detection And Forecasting}},
	author={He, Xiao and Li, Ye and Tan, Jian and Wu, Bin and Li, Feifei},
	journal={Proceedings of the VLDB Endowment},
	volume={16},
	number={06},
	pages={1399--1412},
	year={2023},
	publisher={VLDB Endowment}.
    url={https://www.vldb.org/pvldb/vol16/p1399-he.pdf}
}
```

## Contact
If you have any question, please contact xiao.hx@alibaba-inc.com
