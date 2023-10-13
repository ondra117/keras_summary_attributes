from keras import Model
from keras.utils import io_utils
from keras.utils.layer_utils import get_layer_index_bound_by_layer_name, count_params
from abc import ABC

class Attribute(ABC):
    attribute_name = ...
    end_attribute_name = ...
    line_lenght = ...
    attribute_compute = ...
    end_attribute_compute = ...
    def __init__(self):
        att_name = self.attribute_name != ...
        self._att_name = att_name
        end_att_name = self.end_attribute_name != ...
        self._end_att_name = end_att_name
        l_lenght = self.line_lenght != ...
        att_fn = self.attribute_compute != ...
        self._att_fn = att_fn
        end_att_fn = self.end_attribute_compute != ...

        if att_name or l_lenght:
            if not att_name:
                raise TypeError(f"Can't instantiate abstract class {self.__class__.__name__} with abstract attribute attribute_name")
            else:
                assert isinstance(self.attribute_name, str)
            if not l_lenght:
                raise TypeError(f"Can't instantiate abstract class {self.__class__.__name__} with abstract attribute line_lenght")
            else:
                assert isinstance(self.line_lenght, int)
            if not att_fn:
                raise TypeError(f"Can't instantiate abstract class {self.__class__.__name__} with abstract attribute function attribute_compute")
            else:
                assert callable(self.attribute_compute)

        if end_att_name or end_att_fn:
            if not end_att_name:
                raise TypeError(f"Can't instantiate abstract class {self.__class__.__name__} with abstract attribute end_attribute_name")
            else:
                assert isinstance(self.end_attribute_name, str)
            if not end_att_fn:
                raise TypeError(f"Can't instantiate abstract class {self.__class__.__name__} with abstract attribute function end_attribute_compute")
            else:
                assert callable(self.end_attribute_compute)

def print_summary(
    model,
    attributes=[],
    show_trainable=False,
    line_length=None,
    positions=None,
    print_fn=None,
    expand_nested=False,
    layer_range=None,
):
    """Prints a summary of a model.

    Args:
        model: Keras model instance.
        attributes: Listo of attributes to apli.
        show_trainable: Whether to show if a layer is trainable.
            If not provided, defaults to `False`.
        line_length: Total length of printed lines
            (e.g. set this to adapt the display to different
            terminal window sizes).
        positions: Relative or absolute positions of log elements in each line.
            If not provided, defaults to `[.33, .55, .67, 1.]`.
        print_fn: Print function to use.
            It will be called on each line of the summary.
            You can set it to a custom function
            in order to capture the string summary.
            It defaults to `print` (prints to stdout).
        expand_nested: Whether to expand the nested models.
            If not provided, defaults to `False`.
        layer_range: List or tuple containing two strings,
            the starting layer name and ending layer name (both inclusive),
            indicating the range of layers to be printed in the summary. The
            strings could also be regexes instead of an exact name. In this
             case, the starting layer will be the first layer that matches
            `layer_range[0]` and the ending layer will be the last element that
            matches `layer_range[1]`. By default (`None`) all
            layers in the model are included in the summary.
    """
    if print_fn is None:
        print_fn = io_utils.print_msg

    line_length = line_length or 98
    positions = positions or [0.33, 0.55, 0.67, 1.0]
    if positions[-1] <= 1:
        positions = [int(line_length * p) for p in positions]
    # header names for the different log elements
    to_display = ["Layer (type)", "Output Shape", "Param #", "Connected to"]
    relevant_nodes = []
    for v in model._nodes_by_depth.values():
        relevant_nodes += v

    if show_trainable:
        line_length += 11
        positions.append(line_length)
        to_display.append("Trainable")

    for attribute in attributes:
        if attribute._att_name:
            line_length += attribute.line_lenght
            positions.append(line_length)
            to_display.append(attribute.attribute_name)

    layer_range = get_layer_index_bound_by_layer_name(model, layer_range)

    def print_row(fields, positions, nested_level=0):
        left_to_print = [str(x) for x in fields]
        while any(left_to_print):
            line = ""
            for col in range(len(left_to_print)):
                if col > 0:
                    start_pos = positions[col - 1]
                else:
                    start_pos = 0
                end_pos = positions[col]
                # Leave room for 2 spaces to delineate columns
                # we don't need any if we are printing the last column
                space = 2 if col != len(positions) - 1 else 0
                cutoff = end_pos - start_pos - space
                fit_into_line = left_to_print[col][:cutoff]
                # For nicer formatting we line-break on seeing end of
                # tuple/dict etc.
                line_break_conditions = ("),", "},", "],", "',")
                candidate_cutoffs = [
                    fit_into_line.find(x) + len(x)
                    for x in line_break_conditions
                    if fit_into_line.find(x) >= 0
                ]
                if candidate_cutoffs:
                    cutoff = min(candidate_cutoffs)
                    fit_into_line = fit_into_line[:cutoff]

                if col == 0:
                    line += "|" * nested_level + " "
                line += fit_into_line
                line += " " * space if space else ""
                left_to_print[col] = left_to_print[col][cutoff:]

                # Pad out to the next position
                if nested_level:
                    line += " " * (positions[col] - len(line) - nested_level)
                else:
                    line += " " * (positions[col] - len(line))
            line += "|" * nested_level
            print_fn(line)

    print_fn('Model: "{}"'.format(model.name))
    print_fn("_" * line_length)
    print_row(to_display, positions)
    print_fn("=" * line_length)

    def print_layer_summary_with_connections(layer, nested_level=0):
        """Prints a summary for a single layer (including topological connections).

        Args:
            layer: target layer.
            nested_level: level of nesting of the layer inside its parent layer
              (e.g. 0 for a top-level layer, 1 for a nested layer).
        """
        try:
            output_shape = layer.output_shape
        except AttributeError:
            output_shape = "multiple"
        connections = []
        for node in layer._inbound_nodes:
            if relevant_nodes and node not in relevant_nodes:
                # node is not part of the current network
                continue

            for (
                inbound_layer,
                node_index,
                tensor_index,
                _,
            ) in node.iterate_inbound():
                connections.append(
                    "{}[{}][{}]".format(
                        inbound_layer.name, node_index, tensor_index
                    )
                )

        name = layer.name
        cls_name = layer.__class__.__name__
        fields = [
            name + " (" + cls_name + ")",
            output_shape,
            layer.count_params(),
            connections,
        ]

        if show_trainable:
            fields.append("Y" if layer.trainable else "N")

        for attribute in attributes:
            if attribute._att_fn:
                att_out = attribute.attribute_compute(layer)
                if attribute._att_name:
                    fields.append(att_out)

        print_row(fields, positions, nested_level)

    def print_layer(layer, nested_level=0, is_nested_last=False):
        print_layer_summary_with_connections(layer, nested_level)

        if expand_nested and hasattr(layer, "layers") and layer.layers:
            print_fn(
                "|" * (nested_level + 1)
                + "¯" * (line_length - 2 * nested_level - 2)
                + "|" * (nested_level + 1)
            )

            nested_layer = layer.layers
            is_nested_last = False
            for i in range(len(nested_layer)):
                if i == len(nested_layer) - 1:
                    is_nested_last = True
                print_layer(nested_layer[i], nested_level + 1, is_nested_last)

            print_fn(
                "|" * nested_level
                + "¯" * (line_length - 2 * nested_level)
                + "|" * nested_level
            )

        if not is_nested_last:
            print_fn(
                "|" * nested_level
                + " " * (line_length - 2 * nested_level)
                + "|" * nested_level
            )

    for layer in model.layers[layer_range[0] : layer_range[1]]:
        print_layer(layer)
    print_fn("=" * line_length)

    if hasattr(model, "_collected_trainable_weights"):
        trainable_count = count_params(model._collected_trainable_weights)
    else:
        trainable_count = count_params(model.trainable_weights)

    non_trainable_count = count_params(model.non_trainable_weights)

    print_fn("Total params: {:,}".format(trainable_count + non_trainable_count))
    print_fn("Trainable params: {:,}".format(trainable_count))
    print_fn("Non-trainable params: {:,}".format(non_trainable_count))
    for attribute in attributes:
        if attribute._end_att_name:
            print_fn(f"{attribute.end_attribute_name}: {attribute.end_attribute_compute(model)}")
    print_fn("_" * line_length)

class ModelAtt(Model):
    def summary(
    self,
    attributes=[],
    show_trainable=False,
    line_length=None,
    positions=None,
    print_fn=None,
    expand_nested=False,
    layer_range=None,
    ):
        """Prints a string summary of the network.

        Args:
            attributes: Listo of attributes to apli.
            show_trainable: Whether to show if a layer is trainable.
                If not provided, defaults to `False`.
            line_length: Total length of printed lines
                (e.g. set this to adapt the display to different
                terminal window sizes).
            positions: Relative or absolute positions of log elements
                in each line. If not provided,
                defaults to `[.33, .55, .67, 1.]`.
            print_fn: Print function to use. Defaults to `print`.
                It will be called on each line of the summary.
                You can set it to a custom function
                in order to capture the string summary.
            expand_nested: Whether to expand the nested models.
                If not provided, defaults to `False`.
            layer_range: a list or tuple of 2 strings,
                which is the starting layer name and ending layer name
                (both inclusive) indicating the range of layers to be printed
                in summary. It also accepts regex patterns instead of exact
                name. In such case, start predicate will be the first element
                it matches to `layer_range[0]` and the end predicate will be
                the last element it matches to `layer_range[1]`.
                By default `None` which considers all layers of model.

        Raises:
            ValueError: if `summary()` is called before the model is built.
        """
        if not self.built:
            raise ValueError(
                "This model has not yet been built. "
                "Build the model first by calling `build()` or by calling "
                "the model on a batch of data."
            )
        print_summary(
            self,
            attributes=attributes,
            line_length=line_length,
            positions=positions,
            print_fn=print_fn,
            expand_nested=expand_nested,
            show_trainable=show_trainable,
            layer_range=layer_range,
        )



if __name__ == "__main__":
    from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Activation
    class Net(ModelAtt):
        def __init__(self):
            inp = Input((128, 128, 3))

            x = inp

            x = Conv2D(32, 3, padding="same")(x)

            for _ in range(3):
                skip = x
                x = Conv2D(x.shape[-1], 3, padding="same")(x)
                x = Activation("silu")(x)
                x += skip
                x = Conv2D(x.shape[-1] * 2, 2, strides=2, padding="same")(x)
            x = GlobalAveragePooling2D()(x)
            x = Dense(1)(x)
            super().__init__(inp, x)

    class AttNLayer(Attribute):
        def __init__(self):
            # self.attribute_name = "N Lyaer #"
            self.end_attribute_name = "N Layers"
            # self.line_lenght = 15
            self.idx = 0

            super().__init__()

        def attribute_compute(self, layer):
            self.idx += 1
            # return str(self.idx)
        
        def end_attribute_compute(self, model):
            return str(self.idx)

    a1 = AttNLayer()
    model = Net()
    model.summary(attributes=[a1])