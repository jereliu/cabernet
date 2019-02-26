"""Estimator for sampling model parameters using variational inference (VI)."""
import os
import time
import shutil

import meta.estimator as estimator_template

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed

import util.model as model_util


class VIEstimator(estimator_template.Estimator):
    """Performs VI for a given model by defining tf.Graph."""

    def __init__(self, model):
        """Initializer."""
        super().__init__(model)

    def config(self, step_size=5e-3, **vi_kwargs):
        """Sets up inference graph.

        Args:
            step_size: (float) Step size for optimization.
            **vi_kwargs: Additional keyword arguments for variational family.
        """
        vi_graph = tf.Graph()

        with vi_graph.as_default():
            vi_rv_dict = self.model.variational_family(**vi_kwargs)

            # build likelihood function
            log_likelihood, outcome_rv, model_rv_dict = (
                self._make_likelihood(vi_rv_dict))

            # build kl divergence
            kl = self._make_kl_divergence(vi_rv_dict, model_rv_dict)

            # define loss op by combining likelihood and kl
            # i.e. ELBO = E_q(p(x|z)) + KL(q || p)
            elbo = tf.reduce_mean(log_likelihood - kl)
            nll_val = tf.reduce_mean(-log_likelihood)
            kl_val = tf.reduce_mean(kl)

            loss_op = -elbo

            # define training op
            optimizer = tf.train.AdamOptimizer(step_size)
            train_op = optimizer.minimize(loss_op)

            # define summary op
            tf.summary.scalar("loss", loss_op)
            tf.summary.scalar("neg_likelihood", nll_val)
            tf.summary.scalar("kl_divergence", kl_val)

            summary_op = tf.summary.merge_all()

            # define init op
            init_op = tf.global_variables_initializer()

            # define saving op
            save_op = tf.train.Saver()

        self.graph = vi_graph
        self.param = self.model.vi_param
        self.ops = estimator_template.EstimatorOps(loss=loss_op,
                                                   train=train_op,
                                                   init=init_op,
                                                   save=save_op,
                                                   pred=outcome_rv,
                                                   summary=summary_op)

    def run(self, sess=None,
            max_steps=10000, model_dir=None,
            save_step=500, verbose=False):
        """Executes estimator graph in a given session.

        Args:
            sess: (tf.Session) A session to run graph in.
            max_steps: (int) number of training iterations.
            model_dir: (str) directory for model checkpoints.
            save_step: (int) number of iteration to print output.
            verbose: (bool) whether to print to additional message to stdout.

        Returns:
            sess: (tf.Session) Session after graph evaluation that
                contains estimated parameters.
        """

        if not self.ops or not self.graph or not self.param:
            raise ValueError("Model not properly initialized. "
                             "Please run config()")

        if not model_dir:
            model_dir = os.path.join(os.getcwd(), "ckpt_temp")
            shutil.rmtree(model_dir, ignore_errors=True)
            os.makedirs(model_dir, exist_ok=True)

            print("'model_dir' empty."
                  " Temporary address created: {}".format(model_dir))

        if not sess:
            sess = tf.Session(graph=self.graph)

        # setup summary
        summary_writer = tf.summary.FileWriter(model_dir, self.graph)

        # setup initialization
        sess.run(self.ops.init)

        # execute training
        start_time = time.time()
        step = 0
        try:
            for step in range(max_steps):
                _, elbo_value = sess.run([self.ops.train,
                                          self.ops.loss])
                if step % save_step == 0:
                    # update summary
                    summary_value = sess.run(self.ops.summary)
                    summary_writer.add_summary(summary_value, step)
                    summary_writer.flush()

                    # save model
                    checkpoint_file = os.path.join(model_dir, "model.ckpt")
                    self.ops.save.save(sess, checkpoint_file,
                                       global_step=step)

                    # optionally, print to stdout
                    if verbose:
                        duration = time.time() - start_time
                        print("Step: {:>3d} Loss: {:.3f} ({:.3f} min)".format(
                            step, elbo_value, duration / 60.))
        except KeyboardInterrupt:
            print("================================\n"
                  "Training terminated at Step {}\n"
                  "================================".format(step))

        return sess

    def _make_likelihood(self, rv_dict):
        """Produces optimizable tensor for model likelihood.

        Args:
            rv_dict: (dict of RandomVariable) Dictionary of random variables
                representing variational family for each model parameter.

        Returns:
            log_likelihood: (tf.Tensor) A likelihood tensor with registered
                gradient with respect to VI parameters.
            outcome_rv: (ed.RandomVariable) A random variable representing
                model's predictive distribution.
            model_tape: (ContextManager) A ContextManager recording the
                model variables in model graph.
        """
        with ed.tape() as model_tape:
            with ed.interception(model_util.make_value_setter(**rv_dict)):
                outcome_rv = self.model.definition()

        log_likelihood = self.model.likelihood(outcome_rv, self.model.y)

        return log_likelihood, outcome_rv, model_tape

    def _make_kl_divergence(self, rv_dict, model_tape):
        """Produces optimizable tensor for KL divergence.

        Args:
            rv_dict: (dict of RandomVariable) Dictionary of random variables
                representing variational family for each model parameter.
            model_tape: (ContextManager) A ContextManager recording the
                model variables in model graph.

        Returns:
            (tf.Tensor) A tensor representing KL divergence between
                model and variational parameters
        """
        kl = 0.
        for rv_name, vi_rv in rv_dict.items():
            # compute analytical form
            param_kl = vi_rv.distribution.kl_divergence(
                model_tape[rv_name].distribution)

            kl += tf.reduce_sum(param_kl)

        return kl
