import tensorflow as tf
from tensorflow.python.keras.engine import compile_utils, data_adapter

from losses import MeanSquaredError


class LevenbergMarquardt:
    def __init__(self, model, optimizer=tf.keras.optimizers.SGD(learning_rate=1.0), loss=MeanSquaredError(),
                 attempts_per_step=10, experimental_use_pfor=True, starting_value=0.5, dec_factor=0.2,
                 inc_factor=5., min_value=1e-10, max_value=1e+10, patience=100):
        """
        Parameters
        ----------
        model : tf.keras.Model
        optimizer: (Optional) Performs the update of the model trainable variables. When tf.keras.optimizers.SGD
            is used it is equivalent to the operation `w = w - learning_rate * updates`, where updates is
            the step computed using the Levenberg-Marquardt algorithm.
        loss: (Optional) An object which inherits from tf.keras.losses.Loss and have an additional function to
            compute residuals.
        attempts_per_step: Integer (Optional) During the train step when new model variables are computed,
            the new loss is evaluated and compared with the old loss value. If new_loss < loss,
            then the new variables are accepted, otherwise the old variables are restored and
            new ones are computed using a different damping-factor. This argument represents the
            maximum number of attempts, after which the step is taken.
        experimental_use_pfor: (Optional) If true, vectorizes the jacobian computation. Else falls back to a
            sequential while_loop. Vectorization can sometimes fail or lead to excessive memory usage.
            This option can be used to disable vectorization in such cases.
        starting_value: (Optional) Used to initialize the Trainer internal damping_factor.
        dec_factor: (Optional) Used in the train_step decrease the damping_factor when new_loss < loss.
        inc_factor: (Optional) Used in the train_step increase the damping_factor when new_loss >= loss.
        min_value: (Optional) Used as a lower bound for the damping_factor. Higher values improve numerical
            stability in the resolution of the linear system, at the cost of slower convergence.
        max_value: (Optional) Used as an upper bound for the damping_factor, and as condition to stop the
            Training process.
        patience: (Optional) Number of steps to continue training after max_value has been reached.
        """
        if not model.built:
            raise ValueError("""Trainer model has not yet been built. Build the model first by calling `build()` or 
            calling `fit()` with some data, or specify an `input_shape` argument in the first layer(s) for 
            automatic build.""")

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.attempts_per_step = attempts_per_step
        self.experimental_use_pfor = experimental_use_pfor

        self.starting_value = starting_value
        self.dec_factor = dec_factor
        self.inc_factor = inc_factor
        self.min_value = min_value
        self.max_value = max_value
        self.patience = patience

        self.will_stop = False

        def solve(matrix, rhs):
            return tf.linalg.solve(matrix, rhs)

        self.solve_function = solve

        # Keep track of the current damping_factor.
        self.damping_factor = tf.Variable(self.starting_value, trainable=False, dtype=self.model.dtype)
        self.steps_after_patience = tf.Variable(0, trainable=False)

        # Used to back up and restore model variables.
        self._backup_variables = []

        # Since training updates are computed with shape (num_variables, 1),
        # self._splits and self._shapes are needed to split and reshape the
        # updates so that they can be applied to the model trainable_variables.
        self._splits = []
        self._shapes = []

        for variable in self.model.trainable_variables:
            variable_shape = tf.shape(variable)
            variable_size = tf.reduce_prod(variable_shape)
            backup_variable = tf.Variable(tf.zeros_like(variable), trainable=False)

            self._backup_variables.append(backup_variable)
            self._splits.append(variable_size)
            self._shapes.append(variable_shape)

        self._num_variables = tf.reduce_sum(self._splits).numpy().item()
        self._num_outputs = None

    def init_step(self, damping_factor, loss):
        return damping_factor

    def decrease(self, damping_factor, loss):
        return tf.math.maximum(damping_factor * self.dec_factor, self.min_value)

    def increase(self, damping_factor, loss):
        return tf.math.minimum(damping_factor * self.inc_factor, self.max_value)

    def stop_training(self, damping_factor, loss):
        self.steps_after_patience.assign_add(1)
        if damping_factor >= self.max_value and not self.will_stop:
            self.will_stop = True
            tf.print(f"Max value reached, will stop in {self.patience} steps.")
        return (damping_factor >= self.max_value) and (self.steps_after_patience >= self.patience)

    def apply(self, damping_factor, JJ):
        damping = tf.eye(tf.shape(JJ)[0], dtype=JJ.dtype)
        damping = tf.scalar_mul(damping_factor, damping)
        return tf.add(JJ, damping)

    @tf.function
    def _compute_jacobians(self, inputs, targets):
        with tf.GradientTape(persistent=True) as tape:
            outputs = self.model(inputs, training=True)
            targets, outputs, _ = compile_utils.match_dtype_and_rank(targets, outputs, None)
            residuals = self.loss.residuals(targets, outputs)

        jacobians = tape.jacobian(residuals, self.model.trainable_variables,
                                  experimental_use_pfor=self.experimental_use_pfor,
                                  unconnected_gradients=tf.UnconnectedGradients.ZERO)
        del tape

        num_residuals = tf.size(residuals)
        jacobians = [tf.reshape(j, (num_residuals, -1)) for j in jacobians]

        return jacobians, residuals, outputs

    def _init_gauss_newton(self, inputs, targets):
        # Calculate Jacobians for each layer and Quasi-Hessian matrices
        Js, residuals, outputs = self._compute_jacobians(inputs, targets)
        # print("Residuals")
        # print(residuals.shape)
        # print("\n")

        JJs = [tf.linalg.matmul(J, J, transpose_a=True) for J in Js]
        rhs = [tf.linalg.matmul(J, residuals, transpose_a=True) for J in Js]
        return Js, JJs, rhs, outputs

    # def _compute_gauss_newton_overdetermined(self, Js, JJs, rhs):
    # Calculate delta W for each layer of the network
    # updates = [self.solve_function(JJ, rhs) for JJ in JJs]
    # return updates

    def _compute_gauss_newton_underdetermined(self, Js, JJs, rhs):
        # Calculate delta W for each layer of the network
        updates = [self.solve_function(JJ, rh) for JJ, rh in zip(JJs, rhs)]
        return updates

    def _train_step(self, inputs, targets, compute_gauss_newton):
        Js, JJs, rhs, outputs = self._init_gauss_newton(inputs, targets)

        # print("Jacobians")
        # print([J.shape for J in Js])
        # print("\n")

        # print("JJs")
        # print([JJ.shape for JJ in JJs])
        # print("\n")

        # print("RHS")
        # print([rh.shape for rh in rhs])
        # print("\n")

        # Compute the current loss value.
        loss = self.loss(targets, outputs)

        stop_training = False
        attempt = 0
        damping_factor = self.init_step(self.damping_factor, loss)

        attempts = tf.constant(self.attempts_per_step, dtype=tf.int32)

        while tf.constant(True, dtype=tf.bool):
            update_computed = False
            try:
                # Apply the damping to the gauss-newton hessian approximation.
                JJs_damped = [self.apply(damping_factor, JJ) for JJ in JJs]
                # print("JJs Damped")
                # print([JJd.shape for JJd in JJs_damped])
                # print("\n")
                # Compute the updates:
                # overdetermined: updates = (J' * J + damping)^-1* J' * residuals
                # under determined: updates = J' * (J*J' + damping)^-1 * residuals
                updates = compute_gauss_newton(Js, JJs_damped, rhs)
                # print("Updates")
                # print([update.shape for update in updates])
                # print("\n")
            except Exception as e:
                print("Exception: ", e)
                del e
            else:
                finite = [tf.reduce_all(tf.math.is_finite(update)) for update in updates]
                if tf.reduce_all(finite):
                    update_computed = True
                    # Split and Reshape the updates
                    updates = [tf.reshape(update, shape) for update, shape in zip(updates, self._shapes)]

                    # Apply the updates to the model trainable_variables.
                    self.optimizer.apply_gradients(zip(updates, self.model.trainable_variables))

            if attempt < attempts:
                attempt += 1

                if update_computed:
                    # Compute the new loss value.
                    outputs = self.model(inputs, training=False)
                    new_loss = self.loss(targets, outputs)

                    if new_loss < loss:
                        # Accept the new model variables and backup them.
                        loss = new_loss
                        damping_factor = self.decrease(damping_factor, loss)
                        self.backup_variables()
                        break

                    # Restore the old variables and try a new damping_factor.
                    self.restore_variables()

                damping_factor = self.increase(
                    damping_factor, loss)

                # TODO: Removed stop training
                stop_training = self.stop_training(
                    damping_factor, loss)
                if stop_training:
                    break
            else:
                break

        # Update the damping_factor which will be used in the next train_step.
        self.damping_factor.assign(damping_factor)
        return loss, outputs, attempt, stop_training

    def _compute_num_outputs(self, inputs, targets):
        input_shape = inputs.shape[1::]
        target_shape = targets.shape[1::]
        _inputs = tf.keras.Input(shape=input_shape, dtype=inputs.dtype)
        _targets = tf.keras.Input(shape=target_shape, dtype=targets.dtype)
        outputs = self.model(_inputs)
        _targets, outputs, _ = compile_utils.match_dtype_and_rank(_targets, outputs, None)
        residuals = self.loss.residuals(_targets, outputs)
        return tf.reduce_prod(residuals.shape[1::])

    def reset_damping_factor(self):
        self.steps_after_patience.assign(0)
        self.damping_factor.assign(self.starting_value)

    def backup_variables(self):
        zip_args = (self.model.trainable_variables, self._backup_variables)
        for variable, backup in zip(*zip_args):
            backup.assign(variable)

    def restore_variables(self):
        zip_args = (self.model.trainable_variables, self._backup_variables)
        for variable, backup in zip(*zip_args):
            variable.assign(backup)

    def train_step(self, inputs, targets):
        if self._num_outputs is None:
            self._num_outputs = self._compute_num_outputs(inputs, targets)

        batch_size = tf.shape(inputs)[0]
        num_residuals = batch_size * self._num_outputs
        overdetermined = num_residuals >= self._num_variables

        if overdetermined:
            loss, outputs, attempts, stop_training = self._train_step(inputs, targets,
                                                                      self._compute_gauss_newton_underdetermined)
        else:
            loss, outputs, attempts, stop_training = self._train_step(inputs, targets,
                                                                      self._compute_gauss_newton_underdetermined)

        return loss, outputs, attempts, stop_training

    def fit(self, dataset, epochs=1, metrics=None):
        """Trains self.model on the dataset for a fixed number of epochs.

        Arguments:
            dataset: A `tf.data` dataset, must return a tuple (inputs, targets).
            epochs: Integer. Number of epochs to train the model.
            metrics: List of metrics to be evaluated during training.
        """
        self.backup_variables()
        steps = dataset.cardinality().numpy().item()
        stop_training = False

        if metrics is None:
            metrics = []

        pl = tf.keras.callbacks.ProgbarLogger(count_mode='steps',
                                              stateful_metrics=["damping_factor", "attempts"])

        pl.set_params({"verbose": 1, "epochs": epochs, "steps": steps})

        pl.on_train_begin()

        for epoch in range(epochs):
            if stop_training:
                break

            # Reset metrics.
            for m in metrics:
                m.reset_states()

            pl.on_epoch_begin(epoch)

            iterator = iter(dataset)

            for step in range(steps):
                if stop_training:
                    break

                pl.on_train_batch_begin(step)

                data = next(iterator)

                data = data_adapter.expand_1d(data)
                inputs, targets, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

                loss, outputs, attempts, stop_training = self.train_step(inputs, targets)

                # Update metrics.
                for m in metrics:
                    m.update_state(targets, outputs)

                logs = {
                    "damping_factor": self.damping_factor,
                    "attempts": attempts,
                    "loss": loss,
                }
                logs.update({m.name: m.result() for m in metrics})

                pl.on_train_batch_end(step, logs)

            pl.on_epoch_end(epoch)

        pl.on_train_end()


class LMWrapper(tf.keras.Model):
    def __init__(self, model):
        super(LMWrapper, self).__init__()
        self.model = model
        self.trainer = None

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training, mask)

    def compile(self, optimizer=tf.keras.optimizers.SGD(learning_rate=1.0), loss=MeanSquaredError(),
                attempts_per_step=10, experimental_use_pfor=True, metrics=None, loss_weights=None,
                weighted_metrics=None, **kwargs):
        super(LMWrapper, self).compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights,
                                       weighted_metrics=weighted_metrics, run_eagerly=True)

        self.built = self.model.built

        self.trainer = LevenbergMarquardt(model=self,
                                          optimizer=optimizer,
                                          loss=loss,
                                          attempts_per_step=attempts_per_step,
                                          experimental_use_pfor=experimental_use_pfor)

    def train_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        loss, y_pred, attempts, stop_training = self.trainer.train_step(x, y)

        logs = {
            "damping_factor": self.trainer.damping_factor,
            "attempts": attempts,
            "loss": loss,
        }

        self.compiled_metrics.update_state(y, y_pred)
        self._validate_target_and_loss(y, loss)

        metrics = self.compute_metrics(x, y, y_pred, sample_weight)

        if 'loss' in metrics:
            del metrics['loss']

        logs.update(metrics)
        self.stop_training = stop_training
        return logs

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, **kwargs):
        if verbose > 0:
            if callbacks is None:
                callbacks = []

            # Include attempts to update weights and the damping factor
            logger = tf.keras.callbacks.ProgbarLogger(count_mode='steps',
                                                      stateful_metrics=["damping_factor", "attempts"])
            callbacks.append(logger)

        return super(LMWrapper, self).fit(x=x, y=y, batch_size=batch_size, epochs=epochs,
                                          verbose=verbose, callbacks=callbacks, **kwargs)
