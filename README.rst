====
Hawk
====


.. image:: https://img.shields.io/pypi/v/hawk.svg
        :target: https://pypi.python.org/pypi/hawk

.. image:: https://github.com/PaoloBonettiPolimi/hawk/actions/workflows/main.yml/badge.svg
        :target: https://github.com/PaoloBonettiPolimi/hawk/actions/workflows/main.yml

.. image:: https://readthedocs.org/projects/hawk/badge/?version=latest
        :target: https://hawk.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/github/license/PaoloBonettiPolimi/hawk.svg
    :target: https://github.com/PaoloBonettiPolimi/hawk/blob/master/LICENSE.txt
    :alt: GitHub license

.. image:: https://badges.gitter.im/bird-house/birdhouse.svg
    :target: https://gitter.im/bird-house/birdhouse?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
    :alt: Join the chat at https://gitter.im/bird-house/birdhouse

Hawk (the bird)
  *Hawk is a bird designed to perform causal analysis for climate data or, in general, for time-series.*

The hawk is a raptor with high visual acuity. This prototype takes the name of this bird since the causal analysis that is performed tries to provide an insight on the causal relationship between the variables of a given dataset. 

Hawk is designed to take as input a dataset, divided into train and test set, provided by the user. This prototype then produces a causal analysis as output, which depends on the additional parameters set by the user, analysing the relationship between the features and the target variable of the dataset provided. 

Hawk does not address a specific causal analysis on a specific dataset, but it can be applied to any supervised learning task based on time-series. The current implementation is based on the four classical assumptions of causal discovery: acyclicity, causal sufficiency, faithfulness and Markov assumption. 

The Hawk prototype provides a framework to identify causal relationships in supervised learning settings, which is a generalization of the methodology that has firstly been explored in https://www.politesi.polimi.it/handle/10589/219074. Specifically, that thesis contains additional details on the methodology followed by this prototype and applicative examples, showing causal analyses on some relevant sub-basins of the Po River. 

Hawk is based on two main algorithms: PCMCI and TEFS.

The PCMCI algorithm (J. Runge, P. Nowack, M. Kretschmer, S. Flaxman, D. Sejdinovic, Detecting and quantifying causal associations in large nonlinear time series datasets. Sci. Adv. 5, eaau4996 (2019)) and its variants perform causal inference for time series data.

Specifically, the PCMCI methodology is designed to estimate causal graphs from observational time series, without the possibility of intervention (observational causal discovery). 

The main parameter of this approach is the choice of the (conditional) independence tests to perform, which is exploited by the algorithm to remove as many spurious causal links among variables as possible.The PCMCI approach can be applied in combination with linear or nonlinear independence tests. 
The other main parameter of this method is the time lag which the user considers to be relevant for the current values of the variables. If contemporaneous causal effects are allowed, the methods is called PCMCI+ (and is applied in Hawk).

The main guarantee of the PCMCI method resides in the identification of a partially directed graph, which removes spurious correlations with high probability and that asymptotically converges to the correct causal graph (if the underlying assumptions hold).
On the other hand, its main weakness in a supervised learning setting resides in being agnostic w.r.t. the concept of target, without providing any guarantee on the regression error.

In the adaptation of the original PCMCI implementation within the Hawk prototype, the target variable and each individual feature are considered as input variables. Then, the Hawk runs the PCMCI+ algorithm, obtaining an estimated causal graph. From this output, it considers the autoregressive component of the target variable (if it is identified as significant by the PCMCI+), and the incoming or undirected causal links related to the target variable itself, as the importance of the autoregressive component and the subset of relevant features, respectively.


The second approach implemented in Hawk is the Transfer Entropy Feature selection algorithm (TEFS, "Bonetti, P., Metelli, A. M., & Restelli, M. (2023, October 17). Causal Feature Selection via Transfer Entropy (https://arxiv.org/abs/2310.11059).").

Differently from the PCMCI, the main characteristic of this approach is to identify a set of relevant features, focusing on providing general guarantees on the regression performance.

Specifically, this approach iteratively identifies the most relevant features, filtering the autoregressive component of the target variable to identify the acyclic flow of information. Therefore, the selected causal features are the variables, ranked by importance, that provide additional information on the evolution of the target w.r.t. its previous observed values and the set of already selected relevant features. 

As described in the original paper, assuming an exact estimation of the transfer entropy, it is possible to control the amount of information loss that follows from the reduction of the original features, and discarding features that have zero transfer entropy with the target guarantee no loss of information. However, given the difficulty to correctly extimate the transfer entropy in practice, the Hawk provides a "conservative" output, in the sense that it selects all the features that iteratively provide a positive amount of estimated transfer entropy as causally relevant.

On the other hand, this methodology does not provide asymptotic guarantees on the identification of the correct underlying causal graph, but it only addresses causality in the sense of the classical Granger definition, i.e., a feature is causally relevant for a target variable if it improves the regression performance w.r.t. its autoregressive component and the other features.

More specifically, in line with classical feature selection approaches, the TEFS methodology can be applied in a forward and in a backward manner, with the first that may be preferred for computational and efficiency reasons, which follow from empirical considerations and that have also a theoretical confirmation. Additionally, as for the PCMCI, the choice of the time lag to consider both for the features and the target have an impact on the final output, since they balance the amount of past timesteps that the user considers to be relevant for the actual target. In the Hawk prototype it is therefore possible to select the maximum time lag to consider for features, target, and to include contemporaneous causal effects.

Finally, a wrapper analysis based on transfer entropy and linear regression is additionally available in the Hawk outputs. Again, this analysis iteratively identifies the most important features in a forward manner, focusing on transfer entropy. However, rather than automatically stop when no information is added by the further selection of a feature, this analysis continues the selection, up to selecting the full set of initial features. Then, the output plot provides the score, in terms of validation coefficient of determination of linear regression. This way, the user can ponder between focusing on transfer entropy for the selection, which is a filter method independent from the subsequent model, and the interest of achieving the best predictive performance. 

Documentation
-------------

Learn more about Hawk in its official documentation at https://clint-hawk.readthedocs.io.

Submit bug reports, questions and feature requests at https://github.com/PaoloBonettiPolimi/hawk/issues

Contributing
------------

You can find information about contributing in our `Developer Guide`_.

Please use bump2version_ to release a new version.


License
-------

* Free software: GNU General Public License v3
* Documentation: https://clint-hawk.readthedocs.io.


Credits
-------

This package was created with Cookiecutter_ and the `bird-house/cookiecutter-birdhouse`_ project template.

The two Python libraries TEFS_ and tigramite_ have been exploited to perform the causal analysis of this bird.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`bird-house/cookiecutter-birdhouse`: https://github.com/bird-house/cookiecutter-birdhouse
.. _`Developer Guide`: https://hawk.readthedocs.io/en/latest/dev_guide.html
.. _bump2version: https://hawk.readthedocs.io/en/latest/dev_guide.html#bump-a-new-version
.. _tigramite: https://github.com/jakobrunge/tigramite
.. _TEFS: https://github.com/teobucci/tefs
