import numpy as np
#Custom Imputer that takes the most frequent value in series ,if the data type was Object, and fill N/A values with this value
class CustomImputer():

    def fit(self, X):
        if   X.dtype == np.dtype('O'): self.fill = X.value_counts().index[0]
        else                            : self.fill = X.mean()
        return self

    def transform(self, X):
        return X.fillna(self.fill)
