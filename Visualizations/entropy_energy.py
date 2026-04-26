from Implementations.utility import (
    get_model_rep,
    load_model,
    show_layer_dirichlet_energy,
    show_layer_rep_entropy,
)
from Model_factory.model import Model


def representation_entropy(
    sol_path,
    device,
    prop_layer,
    num_class,
    drop_probs,
    use_activations,
    conv_methods,
    data,
    nbins,
):
    model = load_model(
        Model,
        sol_path,
        device,
        prop_layer,
        num_class,
        drop_probs,
        use_activations,
        conv_methods,
    )
    conv_representation, act_representation = get_model_rep(model, data)

    return show_layer_rep_entropy(conv_representation, act_representation, data, nbins)


def dirichlet_energy(
    sol_path,
    device,
    prop_layer,
    num_class,
    drop_probs,
    use_activations,
    conv_methods,
    data,
    normalized=False,
):
    model = load_model(
        Model,
        sol_path,
        device,
        prop_layer,
        num_class,
        drop_probs,
        use_activations,
        conv_methods,
    )
    conv_representation, act_representation = get_model_rep(model, data)

    return show_layer_dirichlet_energy(
        conv_representation,
        act_representation,
        data,
        normalized,
    )
