# This is my my my comment

import warnings

import numpy as np
from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ridge_regression
from sklearn.utils.validation import check_is_fitted
import numpy as np
from scipy.optimize import minimize
import time

def custom_ridge_obj(w, x, y, alpha, F, beta):
    loss = np.sum((y - x @ w)**2) + alpha * np.sum(w**2) #+ beta * np.sum(np.abs(w))  # Additional penalty

    if beta > 0:
       loss += beta * np.sum((F*w)**2)  
    return loss

def custom_ridge_obj2(w, x, y, alpha, F, beta):
    loss = np.sum((y - x @ w)**2) #+ beta * np.sum(np.abs(w))  # Additional penalty
    
    if np.max(F)==0:
        loss += alpha * np.sum(w**2)
    else:
        if beta > 0:
           loss += ((alpha + beta)/2.0) * np.sum((F*w)**2)
    
    

    return loss

def custom_ridge_gradient(w, x, y, alpha, F, beta):
    # Calculate the gradient of the loss function
    residuals = y - x @ w
    gradient = -2 * x.T @ residuals + 2 * ((alpha + beta)/2.0) * F**2 * w
    return gradient

def custom_ridge_regression(x, y, alpha, F, beta):
    # Rescale X by 1/F
    x_scaled = x / F

    coef = ridge_regression(x_scaled, y, alpha)
    # Reverse the scaling for the coefficients
    coef /= F
    return coef


from .base import BaseOptimizer


class STLSQG(BaseOptimizer):
    """Sequentially thresholded least squares algorithm.
    Defaults to doing Sequentially thresholded Ridge regression.

    Attempts to minimize the objective function
    :math:`\\|y - Xw\\|^2_2 + \\alpha \\|w\\|^2_2`
    by iteratively performing least squares and masking out
    elements of the weight array w that are below a given threshold.

    See the following reference for more details:

        Brunton, Steven L., Joshua L. Proctor, and J. Nathan Kutz.
        "Discovering governing equations from data by sparse
        identification of nonlinear dynamical systems."
        Proceedings of the national academy of sciences
        113.15 (2016): 3932-3937.

    Parameters
    ----------
    threshold : float, optional (default 0.1)
        Minimum magnitude for a coefficient in the weight vector.
        Coefficients with magnitude below the threshold are set
        to zero.

    alpha : float, optional (default 0.05)
        Optional L2 (ridge) regularization on the weight vector.

    max_iter : int, optional (default 20)
        Maximum iterations of the optimization algorithm.

    ridge_kw : dict, optional (default None)
        Optional keyword arguments to pass to the ridge regression.

    fit_intercept : boolean, optional (default False)
        Whether to calculate the intercept for this model. If set to false, no
        intercept will be used in calculations.

    normalize_columns : boolean, optional (default False)
        Normalize the columns of x (the SINDy library terms) before regression
        by dividing by the L2-norm. Note that the 'normalize' option in sklearn
        is deprecated in sklearn versions >= 1.0 and will be removed.

    copy_X : boolean, optional (default True)
        If True, X will be copied; else, it may be overwritten.

    initial_guess : np.ndarray, shape (n_features) or (n_targets, n_features),
            optional (default None)
        Initial guess for coefficients ``coef_``.
        If None, least-squares is used to obtain an initial guess.

    verbose : bool, optional (default False)
        If True, prints out the different error terms every iteration.

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Weight vector(s).

    ind_ : array, shape (n_features,) or (n_targets, n_features)
        Array of 0s and 1s indicating which coefficients of the
        weight vector have not been masked out, i.e. the support of
        ``self.coef_``.

    history_ : list
        History of ``coef_``. ``history_[k]`` contains the values of
        ``coef_`` at iteration k of sequentially thresholded least-squares.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.integrate import odeint
    >>> from pysindy import SINDy
    >>> from pysindy.optimizers import STLSQ
    >>> lorenz = lambda z,t : [10*(z[1] - z[0]),
    >>>                        z[0]*(28 - z[2]) - z[1],
    >>>                        z[0]*z[1] - 8/3*z[2]]
    >>> t = np.arange(0,2,.002)
    >>> x = odeint(lorenz, [-8,8,27], t)
    >>> opt = STLSQ(threshold=.1, alpha=.5)
    >>> model = SINDy(optimizer=opt)
    >>> model.fit(x, t=t[1]-t[0])
    >>> model.print()
    x0' = -9.999 1 + 9.999 x0
    x1' = 27.984 1 + -0.996 x0 + -1.000 1 x1
    x2' = -2.666 x1 + 1.000 1 x0
    """

    def __init__(
        self,
        threshold=0.1,
        alpha=0.05,
        beta = 0.1, # for penalizing using F which comes from the graph
        max_iter=20, # originally 20
        ridge_kw=None,
        normalize_columns=False,
        fit_intercept=False,
        copy_X=True,
        initial_guess=None,
        verbose=False,
        F_penalize = None
        
    ):
        super(STLSQG, self).__init__(
            max_iter=max_iter,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            normalize_columns=normalize_columns,
        )

        if threshold < 0:
            raise ValueError("threshold cannot be negative")
        if alpha < 0:
            raise ValueError("alpha cannot be negative")
            
        print("*********************************Graph SINDY***********************") #+

        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta
        self.ridge_kw = ridge_kw
        self.initial_guess = initial_guess
        self.verbose = verbose
        self.F_penalize = F_penalize

    def _sparse_coefficients(self, dim, ind, coef, threshold):
        """Perform thresholding of the weight vector(s)"""
        c = np.zeros(dim)
        c[ind] = coef
        big_ind = np.abs(c) >= threshold
        c[~big_ind] = 0
        return c, big_ind

    def _regress(self, x, y, FA):
        """Perform the ridge regression"""
        kw = self.ridge_kw or {}

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=LinAlgWarning)
            try:
                """
                Minimize Objective function ||y - Xw||^2  +  a||w||^2
                OneNote 10/26/2023
                each state variable is calculated seperately
                w is the coefficients, a vector with length = number of terms (J)
                w here is a column of matrix ksi in OneNote
                y is X_dot, a vector with length = number of samples (n)
                x is Theta(X), a matrix with dimension (n,J) and reduced to (n,J/.)
                every iteration the length of terms is reduced
                
                """
                #coef = ridge_regression(x, y, self.alpha, **kw)
                #print("FA\n", FA) #+
                

                #simple method
                #coef = minimize(custom_ridge_obj2, x0=np.zeros(x.shape[1]), args=(x, y, self.alpha, FA, self.beta)).x # + # below zero for simple ridge, more than zero for cutom ridge
                #time modified method that we calculate gradient manually
                #coef = minimize(custom_ridge_obj2, x0=np.zeros(x.shape[1]), args=(x, y, self.alpha, FA, self.beta),
                #  method='L-BFGS-B', jac=custom_ridge_gradient).x # +
                coef = custom_ridge_regression(x, y, self.alpha, FA, self.beta) # beta <0 simple ridge.



                #print('Coef ridge regression: \n', coef)  # +
                #print('F ridge regression: \n', FA)  # +
                #print('x ridge regression shape: ', x.shape) # +
                #print('y ridge regression shape: ', y.shape) # +
                #print('coef ridge regression shape: ', coef.shape) # +
            except LinAlgWarning:
                # increase alpha until warning stops
                self.alpha = 2 * self.alpha
        self.iters += 1
        return coef

    def _no_change(self): #Checks if the coefficient mask has changed after thresholding to determine convergence
        """Check if the coefficient mask has changed after thresholding"""
        this_coef = self.history_[-1].flatten()
        if len(self.history_) > 1:
            last_coef = self.history_[-2].flatten()
        else:
            last_coef = np.zeros_like(this_coef)
        return all(bool(i) == bool(j) for i, j in zip(this_coef, last_coef))

    def _reduce(self, x, y): #Main iterative loop of the STLSQG algorithm.
                            #Performs ridge regression, thresholding, and coefficient updates.
                            #Iterates until convergence or maximum iterations are reached.
        """Performs at most ``self.max_iter`` iterations of the
        sequentially-thresholded least squares algorithm.

        Assumes an initial guess for coefficients and support are saved in
        ``self.coef_`` and ``self.ind_``.
        """
        if self.initial_guess is not None:
            self.coef_ = self.initial_guess

        ind = self.ind_
        n_samples, n_features = x.shape
        n_targets = y.shape[1]
        n_features_selected = np.sum(ind)
        
        
        #handling the case that there are no F for penalizing
        if self.F_penalize is not None:
            FA = self.F_penalize
        else:
            FA = np.zeros((n_targets, n_features))


        # Print initial values for each term in the optimization
        if self.verbose:
            row = [
                "Iteration",
                "|y - Xw|^2",
                "a * |w|_2",
                "|w|_0",
                "Total error: |y - Xw|^2 + a * |w|_2",
            ]
            print(
                "{: >10} ... {: >10} ... {: >10} ... {: >10}"
                " ... {: >10}".format(*row)
            )

        for k in range(self.max_iter):
            if np.count_nonzero(ind) == 0:
                warnings.warn(
                    "Sparsity parameter is too big ({}) and eliminated all "
                    "coefficients".format(self.threshold)
                )
                coef = np.zeros((n_targets, n_features))
                break

            coef = np.zeros((n_targets, n_features))
            #FA[0,0]= 10000
            for i in range(n_targets):
                if np.count_nonzero(ind[i]) == 0:
                    warnings.warn(
                        "Sparsity parameter is too big ({}) and eliminated all "
                        "coefficients".format(self.threshold)
                    )
                    continue
                coef_i = self._regress(x[:, ind[i]], y[:, i], FA[i,ind[i]]) 
                coef_i, ind_i = self._sparse_coefficients(
                    n_features, ind[i], coef_i, self.threshold
                )
                
                #print('AFTER SPARSE Coef ridge regression: \n', coef_i)  # +
                coef[i] = coef_i # check this
                ind[i] = ind_i

            self.history_.append(coef)
            if self.verbose:
                R2 = np.sum((y - np.dot(x, coef.T)) ** 2)
                L2 = self.alpha * np.sum(coef**2)
                L0 = np.count_nonzero(coef)
                row = [k, R2, L2, L0, R2 + L2]
                print(
                    "{0:10d} ... {1:10.4e} ... {2:10.4e} ... {3:10d}"
                    " ... {4:10.4e}".format(*row)
                )
            if np.sum(ind) == n_features_selected or self._no_change():
                # could not (further) select important features
                break
        else:
            warnings.warn(
                "STLSQG._reduce did not converge after {} iterations.".format(
                    self.max_iter
                ),
                ConvergenceWarning,
            )
            try:
                coef
            except NameError:
                coef = self.coef_
                warnings.warn(
                    "STLSQG._reduce has no iterations left to determine coef",
                    ConvergenceWarning,
                )
                
        self.coef_ = coef
        self.ind_ = ind # this is importent to know which terms are contributing
        #print('Final Coef ridge regression: \n', coef)  # +
        #print('Final Coef ridge regression Shape: \n', coef.shape)  # + # (6, 57) 6 sate variables and 57 terms
        #print('Final ind ridge regression: \n', ind)  # +

    @property
    def complexity(self):
        check_is_fitted(self)

        return np.count_nonzero(self.coef_) + np.count_nonzero(
            [abs(self.intercept_) >= self.threshold]
        )
