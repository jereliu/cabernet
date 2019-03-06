"""Class definition for computing evaluation metrics."""
import os
import sys

import meta.critique as critique_template

import numpy as np

import matplotlib.pyplot as plt


class Visualizer(critique_template.Critique):
    def __init__(self, bne_model, X_valid, y_valid_sample, save_addr):
        """Initializer.

        Args:
            bne_model: (bne.BNE) Fitted result.
            X_valid: (np.ndarray) Validation features, shape (n_obs_valid, n_dim).
            y_valid_sample: (np.ndarray) A large validation sample from the
                data-generation mechanism, shape (n_obs_valid, n_sample_valid)
        """
        self.save_addr = save_addr

        # extract cdfs and medians
        super().__init__(bne_model, X_valid, y_valid_sample)

    def plot_predictive_cdf(self, verbose=True):
        """Plots model CDF against true CDFs."""
        save_addr_cdf_pred = os.path.join(self.save_addr, "cdf_prediction")
        os.makedirs(save_addr_cdf_pred, exist_ok=True)

        if verbose:
            print("Writing plots to {}...".format(save_addr_cdf_pred), end="")
            sys.stdout.flush()

        # plot, CDF distance
        X_valid = self.model_cdf.x
        cdf_yval = self.model_cdf.y_eval

        cdf_calib_raw = np.mean(self.bne_model.random_summarizer.cdf_val, axis=0)
        cdf_calib = self.model_cdf.random
        cdf_orig = self.model_cdf.system
        cdf_data = self.model_cdf.data

        plt.ioff()
        for i in range(cdf_calib.shape[1]):
            plt.plot(cdf_yval[:, i], cdf_orig[:, i])
            plt.plot(cdf_yval[:, i], cdf_calib[:, i])
            plt.plot(cdf_yval[:, i], cdf_calib_raw[:, i])
            plt.plot(cdf_yval[:, i], cdf_data[:, i])

            plt.ylim((-0.05, 1.05))
            plt.xlim((np.min(cdf_yval), np.max(cdf_yval)))
            plt.title("x={:.3f}".format(X_valid[i, 0]))

            l1_orig = np.mean(np.abs(cdf_orig[:, i] - cdf_data[:, i]))
            l1_calib = np.mean(np.abs(cdf_calib[:, i] - cdf_data[:, i]))
            l1_calib_raw = np.mean(np.abs(cdf_calib_raw[:, i] - cdf_data[:, i]))

            plt.legend(("Model CDF, {:.3f}".format(l1_orig),
                        "Calib CDF, {:.3f}".format(l1_calib),
                        "Raw Calib CDF, {:.3f}".format(l1_calib_raw),
                        "True CDF"))

            plt.savefig("{}/{}.png".format(save_addr_cdf_pred, i))
            plt.close()

        plt.ion()

        if verbose:
            print("Done!")

    def plot_inducing_cdf(self, calib_sample_id_induce, verbose=True):
        """Plots inducing CDFs against true CDF.

        Args:
            calib_sample_id_induce: (np.ndarray) id of observations
                in self.valid_sample that are closest to inducing locations.
        """
        save_addr_cdf_induce = os.path.join(self.save_addr, "cdf_induce")
        os.makedirs(save_addr_cdf_induce, exist_ok=True)

        if verbose:
            print("Writing plots to {}...".format(save_addr_cdf_induce), end="")
            sys.stdout.flush()

        # prepare data, shape (n_eval, n_obs_induce)
        cdf_induce = self.bne_model.random_model.empir_cdf
        cdf_induce = cdf_induce.reshape(
            (self.bne_model.random_model.n_eval, -1))

        cdf_data = self.valid_sample

        cdf_yval = self.bne_model.random_model.quant_val

        plt.ioff()
        for induce_id, valid_id in enumerate(calib_sample_id_induce):
            empir_cdf = np.mean(cdf_data[[valid_id], :] <
                                cdf_yval[:, [induce_id]], axis=1)

            plt.plot(cdf_yval[:, induce_id], cdf_induce[:, induce_id])
            plt.plot(cdf_yval[:, induce_id], empir_cdf)
            plt.legend(("Inducing CDF", "True CDF"))
            plt.savefig("{}/{}.png".format(save_addr_cdf_induce, induce_id))
            plt.close()

        plt.ion()

        if verbose:
            print("Done!")
