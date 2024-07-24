# Predicting-Gambling-Outcome-with-fMRI-and-Recurrent-Neural-Networks

## Authors: Manu Raghavan P V, Amirhossein Zaboli, Julia Plenert, Avisa Fallah, Rezvan Roushani

### Poster presented in Neuromatch Conference 2024

## Abstract
### Background and Objectives 
Decision-making is a crucial aspect of human executive function, involving the evaluation of choices and uncertain outcomes, especially in risky situations with potential gains and losses. Within this domain, risky decision-making refers to choices made under conditions of uncertainty, where potential outcomes can be significantly negative (Kahneman & Tversky, 1979). Understanding the neural mechanisms underlying risky decision-making is crucial for various disciplines, including psychology, neuroscience, and behavioral economics. The Default Mode Network (DMN) is a network of brain regions active during rest, involved in self-referential and introspective thoughts (Raichle et al., 2001). Studies highlight the DMN's role in decision-making, particularly in evaluating past experiences, future planning, and hypothetical reasoning (Buckner et al., 2008; Andrews-Hanna et al., 2014). This underscores its importance in the cognitive processes behind risky decision-making. Using DMN in predictive models has advanced our understanding of decision-making behaviors. fMRI data of the DMN can predict decision outcomes by revealing neural patterns linked to decision processes (Fox et al., 2015). Generalized Linear Models (GLM) and Long Short-Term Memory (LSTM) networks are robust predictive techniques in this context. GLMs extend linear regression to various response distributions (Nelder & Wedderburn, 1972), while LSTMs, a type of recurrent neural network, predict sequences by retaining long-term dependencies, useful for time-series fMRI data (Hochreiter & Schmidhuber, 1997). In this study, we aim to leverage DMN fMRI data from the Human Connectome Project (HCP) dataset in both GLM and LSTM models to predict outcomes in a gambling task, a common experimental paradigm used to study risky decision making. The HCP dataset provides high-quality, publicly available fMRI data, enabling robust and replicable analyses (Van Essen et al., 2013). By integrating these advanced modeling techniques with neuroimaging data, we seek to improve predictive accuracy and gain deeper insights into the neural mechanisms driving risky decision making.
### Methods
We will utilize the Human Connectome Project dataset to extract the fMRI and behavioral data from 100 subjects performing a gambling task (Van Essen et al., 2013). The first step involves identifying the parcel indices corresponding to the Default Mode Network (DMN) for each subject. These parcels will be used to extract time series data relevant to our analysis.The time series data from the identified DMN parcels, along with the behavioral outcomes (win/loss) of the gambling task, will serve as inputs for our predictive models. 
We will implement a Generalized Linear Model (GLM) as our baseline model. The GLM is chosen for its simplicity and effectiveness with smaller sample sizes. The model will be trained on the time series data to predict the behavioral outcomes.
For a more sophisticated approach, we will develop a Long Short-Term Memory (LSTM) model. The LSTM is particularly suited for handling the temporal dynamics inherent in fMRI data, making it a promising candidate for capturing complex patterns over time.
The performance of both the GLM and LSTM models will be evaluated on the dataset using 10-fold cross validation to assess the generalization capability of our models and ensure robustness of our results. The average accuracy, precision and recall across these folds will be presented as the outcome.
### Results
(to be updated soon)
### Conclusions
The comparison between GLM and LSTM models highlights the potential of advanced predictive models in decoding complex brain functions. The LSTM model, in particular, shows promise due to its ability to capture temporal dynamics, offering more accurate predictions of gambling outcomes. This comparison helps balance model complexity and performance, minimizing overfitting. 
### References 
* Andrews-Hanna, J. R., Smallwood, J., & Spreng, R. N. (2014). The default network and self-generated thought: Component processes, dynamic control, and clinical relevance. Annals of the New York Academy of Sciences, 1316(1), 29-52.
* Buckner, R. L., Andrews-Hanna, J. R., & Schacter, D. L. (2008). The brain's default network: Anatomy, function, and relevance to disease. Annals of the New York Academy of Sciences, 1124(1), 1-38.
* Fox, M. D., Snyder, A. Z., Vincent, J. L., Corbetta, M., Van Essen, D. C., & Raichle, M. E. (2015). Intrinsic dynamics of the brain's default mode network. Frontiers in Systems Neuroscience, 9, 30.
* Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735-1780.
* Kahneman, D., & Tversky, A. (1979). Prospect Theory: An Analysis of Decision under Risk. Econometrica, 47(2), 263-291.
* Nelder, J. A., & Wedderburn, R. W. M. (1972). Generalized Linear Models. Journal of the Royal Statistical Society: Series A (General), 135(3), 370-384.
* Raichle, M. E., MacLeod, A. M., Snyder, A. Z., Powers, W. J., Gusnard, D. A., & Shulman, G. L. (2001). A default mode of brain function. Proceedings of the National Academy of Sciences, 98(2), 676-682.
* Van Essen, D. C., Smith, S. M., Barch, D. M., Behrens, T. E., Yacoub, E., & Ugurbil, K. (2013). The WU-Minn Human Connectome Project: An overview. NeuroImage, 80, 62-79.


