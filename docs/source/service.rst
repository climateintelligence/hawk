.. _service:

Climate Service
===============

.. contents::
    :local:
    :depth: 1

Scientific Methode
------------------


Hawk is designed to take as input a dataset, divided into train and test set, producing a causal analysis as output, which depends on the additional parameters set by the users.

Hawk does not address a specific causal analysis on a specific dataset, but it can be applied to any supervised learning task based on time-series.

The work described in https://www.politesi.polimi.it/handle/10589/219074 contains additional details and examples of the causal methods contained in Hawk, showing causal analyses on some relevant sub-basins of the Po River.

Hawk is based on two main algorithms: PCMCI and TEFS.

The PCMCI algorithm (J. Runge, P. Nowack, M. Kretschmer, S. Flaxman, D. Sejdinovic, Detecting and quantifying causal associations in large nonlinear time series datasets. Sci. Adv. 5, eaau4996 (2019).) and its variants perform causal inference for time series data.

Specifically, this methodology is designed to estimate causal graphs from observational time series, without the possibility of intervention (observational causal discovery). 

The main parameter of this approach is the choice of the (conditional) independence tests to perform, which is exploited by the algorithm to remove as many spurious causal links among variables as possible.
The PCMCI approach can be applied in combination with linear or nonlinear independence tests. The other main parameter of this method is the time lag which the user considers to be relevant for the current values of the variables. If contemporaneous causal effects are allowed, the methods is called PCMCI+.

The main guarantee of this method resides in the identification of a partially directed graph, which removes spurious correlations with high probability and that asymptotically converges to the correct causal graph (if the underlying assumptions hold).
On the other hand, its main weakness in a supervised learning setting resides in being agnostic w.r.t. the concept of target, without providing any guarantee on the regression error.

In the adaptation of the original PCMCI implementation within the Hawk bird, the target variable has been considered, together with the features, as an individual variable. Therefore, Hawk considers the autoregressive component of the target variable, and the incoming or undirected causal links related to it, to identify the subset of relevant features and the importance of the autoregressive component.


The second approach implemented in Hawk is the Transfer Entropy Feature selection algorithm (TEFS, "Bonetti, P., Metelli, A. M., & Restelli, M. (2023, October 17). Causal Feature Selection via Transfer Entropy (https://arxiv.org/abs/2310.11059).").

Differently from the PCMCI, the main characteristic of this approach is to identify a set of relevant features, focusing on providing general guarantees on the regression performance.

Specifically, this approach iteratively identifies the most relevant features, filtering the autoregressive component of the target variable to identify the acyclic flow of information. Therefore, the selected causal features are the variables, ranked by importance, that provide additional information on the evolution of the target w.r.t. its previous observed values.

On the other hand, this methodology does not provide asymptotic guarantees on the identification of the correct underlying causal graph, but it only addresses causality in the sense of the classical Granger definition, i.e., a feature is causally relevant for a target variable if it improves the regression performance w.r.t. its autoregressive component and the other features.

More specifically, the TEFS methodology can be applied in a forward and in a backward manner, with the first that may be preferred for computational and efficiency reasons. Additionally, as for the PCMCI, the choice of the time lag to consider both for the features and the target have an impact on the final output, since they balance the amount of past timestaps that the user considers to be relevant for the actual target.

The final output of this process is composed of seven files. Three .pdf files contain the images that summarize the results. The first two files show the selected causal features and the related regression scores for the PCMCI and the TEFS, with the different configurations selected in the inputs. Then, the third file shows an image related to the evolution of the regression score by iteratively adding one feature to the set of the selected causal features, which can be seen as a wrapper variation of the TEFS. Finally, four pickle files compose the remaining set of outputs of this process, containing all the relevant details in terms of selected features and regression scores for the TEFS, the PCMCI, the TEFS variation as a wrapper approach, and a set of baselines. 
