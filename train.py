import os
import jax
from flax import nnx
from typing import Any
import optax
import orbax.checkpoint as ocp
from flax.metrics import tensorboard
import grain
from tqdm.auto import tqdm

from .fake_logger import FakeLogger

from .fake_checkpointer import FakeCheckpointer

from pathlib import Path
import jax.numpy as jnp


def train(
    model: nnx.Module,
    optimizer: optax.GradientTransformation,
    train_dataset: grain.IterDataset,
    val_dataset: grain.IterDataset,
    num_train_samples: int,
    num_val_samples: int,
    num_epochs: int,
    batch_size: int,
    rngs: nnx.Rngs = nnx.Rngs(0),
    checkpoint_dir: str | Path | None = None,
    tensorboard_dir: str | Path | None = None,
    restore_from_checkpoint: bool | None = None,
    metadata: dict = {},
) -> tuple[nnx.Module, Any, nnx.Rngs, dict]:
    """
    Trains the given model for a specified number of epochs with mean squared error loss (MSE).

    If there is already a checkpoint for this experiment, it will continue from that checkpoint and load all the states.

    Every epoch, the best model is checkpointed according to the loss function on the validation dataset.
    It stores at max 3 checkpoints with the model state, optimizer state, and metadata that can be used to gain more information.
    Specifically, the model state is the parameters (obtained through nnx.split(model, nnx.Param, nnx.Everything())[1] and optimizer.update(...)),
    the optimizer state is the optax optimizer state (obtained through optimizer.init(params) and optimizer.update(...)),
    and the metadata contains the training and validation loss for that epoch.

    :param model: The model to train, a new instance of the 'Model' class.
    :type model: nnx.Module
    :param optimizer: The optax optimizer to use to train the model (eg. optax.adam)
    :type optimizer: optax.GradientTransformation
    :param train_dataset: A grain iterable dataset that contains the training samples and on iteration returns a dictionary with keys 'features' and 'targets' that correspond to the input features and target outputs respectively
    :type train_dataset: grain.IterDataset
    :param val_dataset: A grain iterable dataset that contains the validation samples and on iteration returns a dictionary with keys 'features' and 'targets' that correspond to the input features and target outputs respectively
    :type val_dataset: grain.IterDataset
    :param num_train_samples: The number of training samples in the training dataset
    :type num_train_samples: int
    :param num_val_samples: The number of validation samples in the validation dataset
    :type num_val_samples: int
    :param num_epochs: The number of epochs to train the model for
    :type num_epochs: int
    :param batch_size: The batch size to use during training
    :type batch_size: int
    :param rngs: The random number generators for all random operations, passed to the model during training as a kwarg 'rngs' and passes a 'train' boolean representing whether or not training is occuring
    :type rngs: nnx.Rngs
    :param checkpoint_dir: Directory to save and restore checkpoints, if none, disables checkpointing (not recommended)
    :type checkpoint_dir: str | Path | None
    :param tensorboard_dir: Directory to save tensorboard logs, if none, disables tensorboard logging (not recommended)
    :type tensorboard_dir: str | Path | None
    :param restore_from_checkpoint: Whether to restore from an existing checkpoint if available
    :type restore_from_checkpoint: bool | None
    :param metadata: Additional metadata to store with checkpoints
    :type metadata: dict
    :return: The trained model after the specified number of epochs, the final optimizer state, the new Rngs, and metadata
    :rtype: tuple[nnx.Module, Any, nnx.Rngs, dict]
    """

    if restore_from_checkpoint is None:
        restore_from_checkpoint = True
    if checkpoint_dir is None:
        restore_from_checkpoint = False

    # Checkpointing setup
    options = ocp.CheckpointManagerOptions(
        max_to_keep=3,
        best_fn=lambda tree: tree["val_loss"] + tree["train_loss"] / 5,
        best_mode="min",
    )  # We want to focus most on the validation loss but if train loss, explodes, it should have an impact
    if checkpoint_dir is None:
        mngr = FakeCheckpointer()
    else:
        mngr = ocp.CheckpointManager(
            checkpoint_dir,
            options=options,
            item_names=("model_state", "optimizer_state", "metadata"),
        )

    # Tensorboard logging
    if tensorboard_dir is None:
        writer = FakeLogger()
    else:
        writer = tensorboard.SummaryWriter(log_dir=str(tensorboard_dir))

    # Get the graphdef (constant), parameters (changing values), and static (everything else)
    graphdef, params, static = nnx.split(model, nnx.Param, nnx.Everything)

    # Get an optimizer state based on the model parameters (params)
    opt_state = optimizer.init(params)  # type: ignore

    # Random state setup
    rngs_graphdef, rngs_state = nnx.split(rngs)
    rngs_leaves, rngs_state_graphdef = jax.tree.flatten(rngs_state)

    # Get the constant graph defs of the params and opt_state for later reconstruction as well as the leaves
    # We can get leaves here because the helper functions only return leaves and we never change the whole param pytree at once
    param_leaves, param_graph_def = jax.tree.flatten(params)
    opt_state_leaves, opt_state_graph_def = jax.tree.flatten(opt_state)

    # Restore checkpoint if it exists
    if restore_from_checkpoint and os.listdir(checkpoint_dir) != []:
        restored = mngr.restore(
            step=None,
            args=ocp.args.Composite(
                model_state=ocp.args.PyTreeRestore(),
                optimizer_state=ocp.args.PyTreeRestore(),
                metadata=ocp.args.JsonRestore(),
            ),
        )

        params = jax.tree.unflatten(
            param_graph_def, jax.tree.leaves(restored["model_state"])
        )
        opt_state = jax.tree.unflatten(
            opt_state_graph_def, jax.tree.leaves(restored["optimizer_state"])
        )
        metadata = restored["metadata"]  # In case you ever use it again

    def loss_fn(
        param_leaves: list[Any],
        x: jax.Array,
        y: jax.Array,
        rngs_leaves: list[Any],
        training: bool,
    ) -> tuple[jax.Array, list[Any]]:
        """
        Calculates the loss of the model on a given batch

        This function uses a few optimization techniques to ensure maximum performance.
        However, this is less noticeable for larger models compared to smaller models.

        1) Firstly, we don't pass in the model as a parameter and instead only pass the changing parameters. As a result, we use nnx.merge to reconstruct the model by capturing its graph definition and static values at the first execution and pass the changed parameters. This can be broken down into a few reasons:
            1) flax.nnx.value_and_grad is not as performant as jax.value_and_grad due to several substeps it takes to ensure proper model state handling and it is written in Python. In smaller models, this overhead is incredibly noticable and therefore something optimize out.
            2) By only passing in the changing parameters, we reduce the amount of data transfer
            3) The value_and_grad function has to flatten the model PyTree in order to be able to pass it to the accelerator which we can minimize by only passing in the parameters
        2) Secondly, we only pass in the parameter leaves of the model instead of the full parameter pytree. Therefore, we reconstruct the parameter PyTree inside the function with one call to jax.tree.unflatten for the following reasons:
            1) This further reduces the amount of data transfer to the accelerator
            2) The value_and_grad function no longer has to flatten AND unflatten the parameter pytree which can be costly and with a static graphdef, we can capture it once to reconstruct the parameters internally. Instead of having to pass both the graph definition and parameters, we rely on the staticness of the graph definition to reconstruct the same pytree with different leaves. A simple workaround for a model that has slightly changing graph definitions would be to pass the graphdef as a static argument.

        So how does the last optimization affect gradient calculation? When gradients are calculated, instead of being in a PyTree, they are just leaves.
        To overcome this, you can easily turn it into a PyTree to use with Optax by using the same graph definition as for the parameters.

        :param param_leaves: The leaves of the pytree representing the model parameters
        :type param_leaves: list[Any]
        :param x: The input features for the batch to be fed directly into the model
        :type x: jax.Array
        :param y: The target predictions for the batch to be fed directly into the loss function
        :type y: jax.Array
        :param rngs_leaves: The leaves of the pytree representing the random number generator state
        :type rngs_leaves: list[Any]
        :param training: Whether or not the model is being trained (affects layers like dropout, batchnorm, etc.)
        :type training: bool
        :return: A loss value for the given batch as the mean squared error as the only element of the array and the new pytree leaves for the rng states
        :rtype: tuple[jax.Array, list[Any]]

        .. warning:: When taking the gradients for the 'param_leaves' arg, the returned gradients are PyTree leaves to be unflattened with the same graphdef as the parameters.
        """

        # Restore the rngs pytree from the leaves
        rngs = nnx.merge(
            rngs_graphdef, jax.tree.unflatten(rngs_state_graphdef, rngs_leaves)
        )

        # Restore the params pytree from the leaves
        params = jax.tree.unflatten(param_graph_def, param_leaves)

        # Restore the model with the current paramsjax.tree.leaves(nnx.split(rngs_graphdef)[1])
        model = nnx.merge(graphdef, params, static)

        # Get model predictions
        preds = model(
            x, rngs=rngs, training=training
        )  # For RNG dependent layers like dropout

        # Get and return the loss value as a jax.Array
        loss = jnp.mean((preds - y) ** 2)

        return loss, jax.tree.leaves(nnx.split(rngs)[1])

    @jax.jit
    def train_step(
        param_leaves: list[Any],
        opt_state_leaves: list[Any],
        x: jax.Array,
        y: jax.Array,
        rngs_leaves: list[Any],
    ) -> tuple[list[Any], list[Any], list[Any], jax.Array]:
        """
        Does one training step on the given inputs

        This function uses a few optimization techniques to ensure maximum performance.
        However, this is less noticeable for larger models compared to smaller models.

        1) Firstly, we don't pass in the model as a parameter and instead only pass the changing parameters. As a result, we use nnx.merge to reconstruct the model by capturing its graph definition and static values at the first execution and pass the changed parameters. This can be broken down into a few reasons:
            1) flax.nnx.jit is not as performant as jax.jit due to several substeps it takes to ensure proper model state handling and it is written in Python. In smaller models, this overhead is incredibly noticable and therefore something optimize out.
            2) By only passing in the changing parameters, we reduce the amount of data transfer
            3) The JIT function has to flatten the model PyTree in order to be able to pass it to the accelerator which we can minimize by only passing in the parameters
        2) Secondly, we only pass in the parameter leaves of the model instead of the full parameter pytree. Therefore, we reconstruct the parameter PyTree inside the function with one call to jax.tree.unflatten for the following reasons:
            1) This further reduces the amount of data transfer to the accelerator
            2) The JIT function no longer has to flatten AND unflatten the parameter pytree which can be costly and with a static graphdef, we can capture it once to reconstruct the parameters internally. Instead of having to pass both the graph definition and parameters, we rely on the staticness of the graph definition to reconstruct the same pytree with different leaves. A simple workaround for a model that has slightly changing graph definitions would be to pass the graphdef as a static argument.
        3) Similarly, we only pass in the optimizer state leaves and not the full PyTree following a similar process as above.

        The last key optimization is the returning of PyTree leaves and not PyTrees for both the model parameters and the optimization state.
        This reduces the overhead of the JIT compiled function by not forcing it to flatten and unflatten the PyTrees.
        For most use cases, this is fine, especially in tight training loops.
        However, you can easily unflatten right outside of this JIT region, or even just return the PyTrees from here.

        :param param_leaves: The leaves of the pytree representing the model parameters
        :type param_leaves: list[Any]
        :param opt_state_leaves: The leaves of the pytree representing the optimizer state
        :type opt_state_leaves: list[Any]
        :param x: The input features for the batch to be fed directly into the model
        :type x: jax.Array
        :param y: The target predictions for the batch to be fed directly into the loss function
        :type y: jax.Array
        :param rngs_leaves: The leaves of the pytree representing the random number generator state
        :type rngs_leaves: list[Any]
        :return: A tuple containing the updated parameter leaves, updated optimizer state leaves, the new random state leaves, and the loss value for the batch
        :rtype: tuple[list[Any], list[Any], list[Any], jax.Array]
        """

        # Restore pytrees from leaves
        params: Any = jax.tree.unflatten(
            param_graph_def, param_leaves
        )  # Convert back into a pytree
        opt_state: optax.OptState = jax.tree.unflatten(
            opt_state_graph_def, opt_state_leaves
        )  # Convert back into a pytree

        (loss, rngs_leaves), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            param_leaves, x, y, rngs_leaves, True
        )  # Get the loss and gradients

        # Update parameters and optimizer state
        params, opt_state = optimizer.update(
            jax.tree.unflatten(param_graph_def, grads), opt_state, params
        )

        # Return updated paramter and optimizer state leaves and the loss value
        return (
            jax.tree.leaves(params),
            jax.tree.leaves(opt_state),
            rngs_leaves,
            loss,
        )

    @jax.jit
    def validation_step(
        param_leaves: list[Any], x: jax.Array, y: jax.Array, rngs_leaves: list[Any]
    ) -> jax.Array:
        """
        Does one validation step on the given inputs

        This function uses a few optimization techniques to ensure maximum performance.
        However, this is less noticeable for larger models compared to smaller models.

        1) Firstly, we don't pass in the model as a parameter and instead only pass the changing parameters. As a result, we use nnx.merge to reconstruct the model by capturing its graph definition and static values at the first execution and pass the changed parameters. This can be broken down into a few reasons:
            1) flax.nnx.jit is not as performant as jax.jit due to several substeps it takes to ensure proper model state handling and it is written in Python. In smaller models, this overhead is incredibly noticable and therefore something optimize out.
            2) By only passing in the changing parameters, we reduce the amount of data transfer
            3) The JIT function has to flatten the model PyTree in order to be able to pass it to the accelerator which we can minimize by only passing in the parameters
        2) Secondly, we only pass in the parameter leaves of the model instead of the full parameter pytree. Therefore, we reconstruct the parameter PyTree inside the function with one call to jax.tree.unflatten for the following reasons:
            1) This further reduces the amount of data transfer to the accelerator
            2) The JIT function no longer has to flatten AND unflatten the parameter pytree which can be costly and with a static graphdef, we can capture it once to reconstruct the parameters internally. Instead of having to pass both the graph definition and parameters, we rely on the staticness of the graph definition to reconstruct the same pytree with different leaves. A simple workaround for a model that has slightly changing graph definitions would be to pass the graphdef as a static argument.
        3) Similarly, we only pass in the optimizer state leaves and not the full PyTree following a similar process as above.

        :param param_leaves: The leaves of the pytree representing the model parameters
        :type param_leaves: list[Any]
        :param x: The input features for the batch to be fed directly into the model
        :type x: jax.Array
        :param y: The target predictions for the batch to be fed directly into the loss function
        :type y: jax.Array
        :param rngs_leaves: The leaves of the pytree representing the random number generator state
        :type rngs_leaves: list[Any]
        :return: The loss value for the batch
        :rtype: jax.Array

        .. note:: This is currently just a JIT wrapper around the loss function but can be extended for more advanced use.
        """
        return loss_fn(param_leaves, x, y, rngs_leaves, False)[0]

    # Calculate number of batches per epoch
    train_steps = num_train_samples // batch_size
    val_steps = num_val_samples // batch_size

    # The actual training loop
    for epoch in range(num_epochs):
        model.train()  # Important for layers like dropout, batchnorm, etc. since we do both training and validation every epoch

        train_loss = 0.0  # Accumulates training loss over the batch

        # Use tqdm for progress bars in the training loop
        for data in tqdm(
            train_dataset,
            total=train_steps,
            desc=f"Train {epoch + 1}/{num_epochs}",
            leave=False,
        ):
            x: jax.Array = jax.device_put(data["features"])  # Move data to accelerator
            y: jax.Array = jax.device_put(data["targets"])  # Move data to accelerator

            # Train
            param_leaves, opt_state_leaves, rngs_leaves, loss = train_step(
                param_leaves, opt_state_leaves, rngs_leaves, x, y
            )

            # Accumulate loss
            train_loss += loss.item()

        train_loss /= train_steps  # Get average training loss

        model.eval()  # Switch to evaluation to disable dropout and other such layers

        val_loss = 0.0  # Accumulates validation loss over the batch

        # Use tqdm for progress bars in the validation loop
        for data in tqdm(
            val_dataset,
            total=val_steps,
            desc=f"Val {epoch + 1}/{num_epochs}",
            leave=False,
        ):
            x: jax.Array = jax.device_put(data["features"])  # Move data to accelerator
            y: jax.Array = jax.device_put(data["targets"])  # Move data to accelerator

            # Validate and accumulate loss
            val_loss += validation_step(param_leaves, x, y).item()

        val_loss /= val_steps  # Get average validation loss

        params = jax.tree.unflatten(
            param_graph_def, param_leaves
        )  # Unflatten for logging
        opt_state = jax.tree.unflatten(
            opt_state_graph_def, opt_state_leaves
        )  # Unflatten for logging

        data = next(
            iter(val_dataset)
        )  # Get a dummy sample of data that (should) be the same every epoch
        x: jax.Array = jax.device_put(data["features"])  # Move data to accelerator
        y: jax.Array = jax.device_put(data["targets"])  # Move data to accelerator

        preds, grads = jax.value_and_grad(loss_fn)(param_leaves, x, y)

        # Tensorboard logging
        for path, grad in jax.tree.flatten_with_path(grads)[0]:
            tag = "/".join([str(p) for p in path])
            writer.histogram(f"gradients/{tag}", grad, step=epoch)

        for path, param in jax.tree.flatten_with_path(params)[0]:
            tag = "/".join([str(p) for p in path])
            writer.histogram(f"parameters/{tag}", param, step=epoch)

        writer.histogram(f"validation/predictions", preds, step=epoch)
        writer.scalar("train/loss", train_loss, epoch)
        writer.scalar("val/loss", val_loss, epoch)

        print(
            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}"
        )

        # Save checkpoint
        mngr.save(
            epoch,
            args=ocp.args.Composite(
                model_state=ocp.args.PyTreeSave(params),
                optimizer_state=ocp.args.PyTreeSave(opt_state),
                metadata=ocp.args.JsonSave(
                    {"train_loss": train_loss, "val_loss": val_loss}
                ),
            ),
            metrics={"train_loss": train_loss, "val_loss": val_loss},
        )

    model = nnx.merge(
        graphdef, params, static
    )  # Merge graphdef, params, and static to return the model

    # Return the final information
    return (
        model,
        opt_state,
        nnx.merge(rngs_graphdef, jax.tree.unflatten(rngs_state_graphdef, rngs_leaves)),
        metadata,
    )
