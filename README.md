# Solving the "Natural Language Processing with Disaster Tweets" Kaggle competition using AdaFlood regulation

Using the "Natural Language Processing with Disaster Tweets" Kaggle competition as a toy problem to save time, I wanted to implement and see how the new [AdaFlood](https://arxiv.org/abs/2311.02891) regularization could help models generalize better. Similar to what they do in the paper, I also implemented [iFlood](https://openreview.net/forum?id=MsHnJPaBUZE) for comparison. 

# What is Flooding?
The idea of *flooding* is to put a lower bound on the loss function that is higher than zero in the hopes of avaoiding overfitting on training samples. The AdaFlood paper mention the first approach to flooding described in [Do We Need Zero Training Loss After Achieving Zero Training Error?](https://arxiv.org/abs/2002.08709), where the lower bound is applied to the whole loss:

$$\mathcal{L}_{Flooding} = |\mathcal{L} - b| + b$$

, where $b$ is the flood level. Unfortunately, this leads to unstable training and and inconsistent results since in practice, $\mathcal{L}$ would be the average over the whole dataset, meaning some datapoints may still have a loss under this new lower bound $b$.

To remedy this caveat, [iFlood](https://openreview.net/forum?id=MsHnJPaBUZE) pushes the idea a bit further and explicitly applies the flood level to all datapoints: 

$$\mathcal{L}_{iFlood} = \frac{1}{B}\sum^B_{i=1}(|\mathcal{l}(y_i, f(\mathbf{x}_i)) - b| + b)$$

, where $B$ is the batch size and $b$ is the flood level. With this formulation, it guarantees that every datapoint has the proper lower bound, but the flood level is still the same for all datapoints. AdaFlood proposes to push things a bit further. Their theory is that easy training samples (i.e. properly labeled and not an outlier) can be driven to zero training loss without hindering performance, but doing the same with bad training samples (i.e. mislabeled, outlier or noisy) will most likely cause overfitting. To prevent this, AdaFlood uses the same local application of the flood level, but it learns the flood level for each datapoint beforehand:

$$\mathcal{L}_{AdaFlood} = \frac{1}{B}\sum^B_{i=1}(|\mathcal{l}(y_i, f(\mathbf{x}_i)) - \theta_i| + \theta_i)$$

Briefly, the weights represent the corrected loss (see equation 5 and 6 in the paper) loss of an auxiliary model. The most simple way to achieve this is to do k-fold cross-validation to train auxiliary models and then compute the loss on the datapoints in the held-out set. Although it is somewhat compute heavy, this only has to be done once. This is how I implemented this in my code.

# Results

| Model | Test F1 |
| --- | --- |
| bert-base-cased | 0.80539 |
| bert-base-cased, adaflood (10 folds, 0.25 gamma, bert-base-cased) | 0.82899 |
| bertweet-base, adaflood (10 folds, 0.25 gamma, bert-base-cased) | 0.83787 |

To really focused on the impact of flooding, I didn't do any preprocessing on the training data. Using a simple BERT model, the model was not so bad considering the leaderboard was showing ~0.85 as the top scores. Then, using 10-fold cross-validation, I learned AdaFlood weights using the same BERT model using a correction gamma of 0.25 (see equation 5 and 6). Just doing that I gained 2.9% nad by then using the same weights but training a bertweet model with these flood levels, I gained an additional 1.1%, landing me the 67th place.

My guess as how this works so well is that the flood levels learned are actually making it so the model ignores the bad datapoints I should've removed as a first preprocessing step. I think doing proper EDA and dataset cleaning could improve the score to be around the ~0.85 mark, but for now I keep this as something to do later since I really wanted to focus on the potential of flooding and better understand how it works.
