import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize

class Signal:
    # Private Method
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------

    def __init__(self, data):

        """
        :param correlation_matrix: Correlation matrix of the set of returns.
        """
        self.data = data
        pass

    def mpPDF(self, var, q, pts):
        """
        Creates a Marchenko-Pastur Probability Density Function
        Args:
            var (float): Variance
            q (float): T/N where T is the number of rows and N the number of columns
            pts (int): Number of points used to construct the PDF
        Returns:
            pd.Series: Marchenko-Pastur PDF
        """
        # Marchenko-Pastur pdf
        # q=T/N
        # Adjusting code to work with 1 dimension arrays
        if isinstance(var, np.ndarray):
            if var.shape == (1,):
                var = var[0]
        eMin, eMax = var * (1 - (1. / q) ** .5) ** 2, var * (1 + (1. / q) ** .5) ** 2
        eVal = np.linspace(eMin, eMax, pts)
        pdf = q / (2 * np.pi * var * eVal) * ((eMax - eVal) * (eVal - eMin)) ** .5
        pdf = pd.Series(pdf, index=eVal)
        return pdf

    def getPCA(self, matrix):
        """
        Gets the Eigenvalues and Eigenvector values from a Hermitian Matrix
        Args:
            matrix pd.DataFrame: Correlation matrix
        Returns:
             (tuple): tuple containing:
                np.ndarray: Eigenvalues of correlation matrix
                np.ndarray: Eigenvectors of correlation matrix
        """
        # Get eVal,eVec from a Hermitian matrix
        eVal, eVec = np.linalg.eigh(matrix)
        indices = eVal.argsort()[::-1]  # arguments for sorting eVal desc
        eVal, eVec = eVal[indices], eVec[:, indices]
        eVal = np.diagflat(eVal)
        return eVal, eVec

    def fitKDE(self, obs, bWidth=.25, kernel='gaussian', x=None):
        """
        Fit kernel to a series of obs, and derive the prob of obs x is the array of values
            on which the fit KDE will be evaluated. It is the empirical PDF
        Args:
            obs (np.ndarray): observations to fit. Commonly is the diagonal of Eigenvalues
            bWidth (float): The bandwidth of the kernel. Default is .25
            kernel (str): The kernel to use. Valid kernels are [‘gaussian’|’tophat’|
                ’epanechnikov’|’exponential’|’linear’|’cosine’] Default is ‘gaussian’.
            x (np.ndarray): x is the array of values on which the fit KDE will be evaluated
        Returns:
            pd.Series: Empirical PDF
        """
        if len(obs.shape) == 1:
            obs = obs.reshape(-1, 1)
        kde = KernelDensity(kernel=kernel, bandwidth=bWidth).fit(obs)
        if x is None:
            x = np.unique(obs).reshape(-1, 1)
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        logProb = kde.score_samples(x)  # log(density)
        pdf = pd.Series(np.exp(logProb), index=x.flatten())
        return pdf

    def errPDFs(self, var, eVal, q, bWidth, pts=1000):
        """
        Fit error of Empirical PDF (uses Marchenko-Pastur PDF)
        Args:
            var (float): Variance
            eVal (np.ndarray): Eigenvalues to fit.
            q (float): T/N where T is the number of rows and N the number of columns
            bWidth (float): The bandwidth of the kernel.
            pts (int): Number of points used to construct the PDF
        Returns:
            float: sum squared error
        """
        # Fit error
        pdf0 = self.mpPDF(var, q, pts)  # theoretical pdf
        pdf1 = self.fitKDE(eVal, bWidth, x=pdf0.index.values)  # empirical pdf
        sse = np.sum((pdf1 - pdf0) ** 2)
        return sse

    def findMaxEval(self, data, bWidth=0.01):
        """
        Find max random eVal by fitting Marchenko’s dist (i.e) everything else larger than
            this, is a signal eigenvalue
        Args:
            data (pd.DataFrame): Time series data
            q (float): T/N where T is the number of rows and N the number of columns
            bWidth (float): The bandwidth of the kernel.
        Returns:
             (tuple): tuple containing:
                float: Maximum random eigenvalue
                float: Variance attributed to noise (1-result) is one way to measure
                    signal-to-noise
        """

        corr0 = data.corr()
        eVal0, eVec0 = self.getPCA(corr0)
        q = data.shape[0] / data.shape[1]

        out = minimize(lambda *x: self.errPDFs(*x), 0.5, args=(np.diag(eVal0), q, bWidth),
                       bounds=((1E-5, 1 - 1E-5),))
        if out['success']:
            var = out['x'][0]
        else:
            var = 1
        eMax = var * (1 + (1. / q) ** .5) ** 2

        nFacts0 = eVal0.shape[0] - np.diag(eVal0)[::-1].searchsorted(eMax)

        return eMax, var, nFacts0


