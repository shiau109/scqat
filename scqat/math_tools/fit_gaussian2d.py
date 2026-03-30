from .function_fitting import FunctionFitting, register_fitter

import numpy as np
from lmfit import Model, Parameters


def gaussian2D_function(x, y, amp, x0, y0, sigma_x, sigma_y):
    return amp * np.exp(
        -( ((x - x0) ** 2) / (2 * sigma_x ** 2) + ((y - y0) ** 2) / (2 * sigma_y ** 2) )
    )


@register_fitter('multi_gaussian2d')
class FitMultiGaussian2D(FunctionFitting):

    def __init__(self, data, x, y, n_gauss=2):
        super().__init__()
        self.data = data
        self.x = x
        self.y = y
        self.n_gauss = n_gauss

        m_offset = Model(lambda x, y, offset: offset * np.ones_like(x), independent_vars=['x', 'y'])
        self.model = m_offset
        for i in range(n_gauss):
            self.model += Model(gaussian2D_function, independent_vars=['x', 'y'], prefix=f'g{i}_')

        self.params = self.guess()

    def guess(self):
        from sklearn.cluster import KMeans
        params = Parameters()
        # Flatten coordinate grids and data
        X, Y = np.meshgrid(self.x, self.y)
        coords = np.column_stack([X.ravel(), Y.ravel()])
        data_flat = self.data.ravel()
        # Use data as weights for clustering
        kmeans = KMeans(n_clusters=self.n_gauss, n_init=10)
        # To avoid NaN, mask out zero/negative density points
        mask = data_flat > 0
        if np.sum(mask) >= self.n_gauss:
            kmeans.fit(coords[mask], sample_weight=data_flat[mask])
            centers = kmeans.cluster_centers_
        else:
            # fallback: uniform grid
            centers = np.column_stack([
                np.linspace(self.x.min(), self.x.max(), self.n_gauss),
                np.linspace(self.y.min(), self.y.max(), self.n_gauss)
            ])
        for i in range(self.n_gauss):
            x0_guess, y0_guess = centers[i]
            params.add(f'g{i}_x0', value=x0_guess)
            params.add(f'g{i}_y0', value=y0_guess)
            params.add(f'g{i}_sigma_x', value=(self.x.max()-self.x.min())/4, min=1e-6)
            params.add(f'g{i}_sigma_y', value=(self.y.max()-self.y.min())/4, min=1e-6)
            # Amplitude guess: max in region near center
            ix = np.abs(self.x - x0_guess).argmin()
            iy = np.abs(self.y - y0_guess).argmin()
            amp_guess = self.data[iy, ix]
            params.add(f'g{i}_amp', value=amp_guess, min=0, vary=True)
        params.add('offset', value=np.min(self.data))
        return params

    def model_function(self, *args, **kwargs):
        pass

    def fit(self):
        result = self.model.fit(
            self.data.ravel(),
            x=np.tile(self.x, len(self.y)),
            y=np.repeat(self.y, len(self.x)),
            params=self.params
        )
        return result


class FitGaussian2D(FunctionFitting):
    def __init__(self, data, x, y):
        self.data = data
        self.x = x
        self.y = y
        self.model = Model(self.model_function, independent_vars=['x', 'y'])
        self.params = self.guess()

    def model_function(self, x, y, amp, x0, y0, sigma_x, sigma_y, offset):
        return gaussian2D_function(x, y, amp, x0, y0, sigma_x, sigma_y) + offset

    def guess(self):
        amp = np.max(self.data) - np.min(self.data)
        offset = np.min(self.data)
        x0 = self.x[np.argmax(np.sum(self.data, axis=0))]
        y0 = self.y[np.argmax(np.sum(self.data, axis=1))]
        sigma_x = (self.x.max() - self.x.min()) / 4
        sigma_y = (self.y.max() - self.y.min()) / 4
        params = Parameters()
        params.add('amp', value=amp)
        params.add('x0', value=x0)
        params.add('y0', value=y0)
        params.add('sigma_x', value=sigma_x, min=0)
        params.add('sigma_y', value=sigma_y, min=0)
        params.add('offset', value=offset)
        return params

    def fit(self):
        result = self.model.fit(
            self.data.ravel(),
            x=np.tile(self.x, len(self.y)),
            y=np.repeat(self.y, len(self.x)),
            params=self.params
        )
        return result
