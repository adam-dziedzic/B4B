# Bucks for Buckets (B4B): Active Defenses Against Stealing Encoders

## TLDR:
We propose the first active defense against encoder extraction that transforms outputs without compromising representation quality for legitimate API users.


## Abstract 

Machine Learning as a Service (MLaaS) APIs provide ready-to-use and high-utility encoders that generate vector representations for given inputs. Since these encoders are very costly to train, they become lucrative targets for model stealing attacks during which an adversary leverages query access to the API to replicate the encoder locally at a fraction of the original training costs. We propose Bucks for Buckets (B4B), the first active defense that prevents stealing while the attack is happening without degrading representation quality for legitimate API users. Our defense relies on the observation that the representations returned to adversaries who try to steal the encoder's functionality cover a significantly larger fraction of the embedding space than representations of legitimate users who utilize the encoder to solve a particular downstream task. B4B leverages this to adaptively adjust the utility of the returned representations according to a user's coverage of the embedding space. To prevent adaptive adversaries from eluding our defense by simply creating multiple user accounts (sybils), B4B also individually transforms each user's representations. This prevents the adversary from directly aggregating representations over multiple accounts to create their stolen encoder copy. Our active defense opens a new path towards securely sharing and democratizing encoders over public APIs.


## Description of the code

The directory `active-defence-for-encoders` contains code for experiments for  mapping, transformations, noising and related evaluation 

The directory `end2end-stealing` contains code for the end-to-end experiment on our defence against encoder stealing. The file `stealsimsiam.py` is used for running the model stealing. The file `linsimsiam.py` is used for evaluation on downstream tasks



