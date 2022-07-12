# FeatureSetEmbedding

This is an implementation of Feature set Embedding introduced in https://papers.nips.cc/paper/2010/hash/5f0f5e5f33945135b874349cfbed4fb9-Abstract.html.

I implemented this approach to handle missing data in a learning problem without the need of imputation. Here each instance is not considered as a vector, rather a set of (feature, value) pair.

Feature set embedding is a two level architecture. At the first level, Set Embedding, the feature value pairs are mapped to an embedding space
of dimension m. At the second level, the embedding vectors are combined using a linear(mean) or a non-linear(max) function to make class predictions.
Detailed information on implementation and design can be found in MasterThesis.pdf.
