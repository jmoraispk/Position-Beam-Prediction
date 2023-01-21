# Position Aided Beam prediction: How useful GPS locations actually are?
This is a python code package related to the following article:

João Morais, and Ahmed Alkhateeb, "[Position Aided Beam prediction: How useful GPS locations actually are?](https://arxiv.org/abs/2205.09054)". (accepted in ICC 2023)

# Instructions to Reproduce the Results

The scripts for generating the results in the paper. The data consists of Scenarios 1-9 of DeepSense6G dataset.

**To reproduce the results, please follow these steps:**
1. Download (or clone) the repository.
2. (optional) Download scenario data & convert to NumPy for fast reading (or use data provided in Gathered data folder)
3. Run 2-normalize_split_train_test_plot.py script
	- change number of beams, number of runs, scenarios, AI approaches, plots, etc.
	- append results in csv file / excel sheel for analysis

Note: a version of the data is already included in the repository, under "Gathered_data_DEV". For downloading the data in the standard format, in case other types of experiments are required, visit the [Scenarios page](https://deepsense6g.net/scenarios/).

If you have any questions regarding the code and used dataset, please write to DeepSense 6G dataset forum https://deepsense6g.net/forum/ or contact [João Morais](mailto:joao@asu.edu?subject=[GitHub]%20Beam%20prediction%20implementation).

For more details, consult the [Position-aided Beam Prediction task](https://deepsense6g.net/position-aided-beam-prediction/) page.

# Abstract of the Article
Millimeter-wave (mmWave) communication systems rely on narrow beams for achieving sufficient receive signal power. Adjusting these beams is typically associated with large training overhead, which becomes particularly critical for highlymobile applications. Intuitively, since optimal beam selection can benefit from the knowledge of the positions of communication terminals, there has been increasing interest in leveraging position data to reduce the overhead in mmWave beam prediction. Prior work, however, studied this problem using only synthetic data that generally does not accurately represent real-world measurements. In this paper, we investigate position-aided beam prediction using a real-world large-scale dataset to derive insights into precisely how much overhead can be saved in practice. Furthermore, we analyze which machine learning algorithms perform best, what factors degrade inference performance in real data, and which machine learning metrics are more meaningful in capturing the actual communication system performance.

# License and Referencing
This code package is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/). 
If you in any way use this code for research that results in publications, please cite our original article:
> > J. Morais, A. Alkhateeb, “Position Aided Beam prediction: How useful GPS locations actually are?” to be available on arXiv, 2022. [Online]. Available: https://arxiv.org/abs/2205.09054

If you use the [DeepSense 6G dataset](www.deepsense6g.net), please also cite our dataset article:
> A. Alkhateeb, G. Charan, T. Osman, A. Hredzak, and N. Srinivas, “DeepSense 6G: large-scale real-world multi-modal sensing and communication datasets,” to be available on arXiv, 2022. [Online]. Available: https://www.DeepSense6G.net
