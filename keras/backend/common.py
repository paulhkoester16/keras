import numpy as np

# the type of float to use throughout the session.
_FLOATX = 'float32'
_EPSILON = 10e-8
_IMAGE_DATA_FORMAT = 'channels_last'


def epsilon():
    """Returns the value of the fuzz factor used in numeric expressions.

    # Returns
        A float.

    # Example
    ```python
        >>> keras.backend.epsilon()
        1e-08
    ```
    """
    return _EPSILON


def set_epsilon(e):
    """Sets the value of the fuzz factor used in numeric expressions.

    # Arguments
        e: float. New value of epsilon.

    # Example
    ```python
        >>> from keras import backend as K
        >>> K.epsilon()
        1e-08
        >>> K.set_epsilon(1e-05)
        >>> K.epsilon()
        1e-05
    ```
    """
    global _EPSILON
    _EPSILON = e


def floatx():
    """Returns the default float type, as a string.
    (e.g. 'float16', 'float32', 'float64').

    # Returns
        String, the current default float type.

    # Example
    ```python
        >>> keras.backend.floatx()
        'float32'
    ```
    """
    return _FLOATX


def set_floatx(floatx):
    """Sets the default float type.

    # Arguments
        floatx: String, 'float16', 'float32', or 'float64'.

    # Example
    ```python
        >>> from keras import backend as K
        >>> K.floatx()
        'float32'
        >>> K.set_floatx('float16')
        >>> K.floatx()
        'float16'
    ```
    """
    global _FLOATX
    if floatx not in {'float16', 'float32', 'float64'}:
        raise ValueError('Unknown floatx type: ' + str(floatx))
    _FLOATX = str(floatx)


def cast_to_floatx(x):
    """Cast a Numpy array to the default Keras float type.

    # Arguments
        x: Numpy array.

    # Returns
        The same Numpy array, cast to its new type.

    # Example
    ```python
        >>> from keras import backend as K
        >>> K.floatx()
        'float32'
        >>> arr = numpy.array([1.0, 2.0], dtype='float64')
        >>> arr.dtype
        dtype('float64')
        >>> new_arr = K.cast_to_floatx(arr)
        >>> new_arr
        array([ 1.,  2.], dtype=float32)
        >>> new_arr.dtype
        dtype('float32')
    ```
    """
    return np.asarray(x, dtype=_FLOATX)


def image_data_format():
    """Returns the default image data format convention ('channels_first' or 'channels_last').

    # Returns
        A string, either `'channels_first'` or `'channels_last'`

    # Example
    ```python
        >>> keras.backend.image_data_format()
        'channels_first'
    ```
    """
    return _IMAGE_DATA_FORMAT


def set_image_data_format(data_format):
    """Sets the value of the data format convention.

    # Arguments
        data_format: string. `'channels_first'` or `'channels_last'`.

    # Example
    ```python
        >>> from keras import backend as K
        >>> K.image_data_format()
        'channels_first'
        >>> K.set_image_data_format('channels_last')
        >>> K.image_data_format()
        'channels_last'
    ```
    """
    global _IMAGE_DATA_FORMAT
    if data_format not in {'channels_last', 'channels_first'}:
        raise ValueError('Unknown data_format:', data_format)
    _IMAGE_DATA_FORMAT = str(data_format)


# Legacy methods

def set_image_dim_ordering(dim_ordering):
    """Legacy setter for `image_data_format`.

    # Arguments
        dim_ordering: string. `tf` or `th`.

    # Example
    ```python
        >>> from keras import backend as K
        >>> K.image_data_format()
        'channels_first'
        >>> K.set_image_data_format('channels_last')
        >>> K.image_data_format()
        'channels_last'
    ```

    # Raises
        ValueError: if `dim_ordering` is invalid.
    """
    global _IMAGE_DATA_FORMAT
    if dim_ordering not in {'tf', 'th'}:
        raise ValueError('Unknown dim_ordering:', dim_ordering)
    if dim_ordering == 'th':
        data_format = 'channels_first'
    else:
        data_format = 'channels_last'
    _IMAGE_DATA_FORMAT = data_format


def image_dim_ordering():
    """Legacy getter for `image_data_format`.

    # Returns
        string, one of `'th'`, `'tf'`
    """
    if _IMAGE_DATA_FORMAT == 'channels_first':
        return 'th'
    else:
        return 'tf'

def _gen_tensor_prod_arg_helper(x_shape, y_shape, reduce_axes=None, elementwise_axes=None):
    """
    Performs argument checking and reformats axes as necessary for gen_tensor_prod
    (defined in the specific backend scripts
    """

    if (reduce_axes is None) or (reduce_axes == []):
        reduce_axes = []
    elif isinstance(reduce_axes, int):
        reduce_axes = [(reduce_axes, reduce_axes)]
    elif isinstance(reduce_axes, list) and isinstance(reduce_axes[0], int):
        reduce_axes = [(r, r) for r in reduce_axes]

    for (i, j) in reduce_axes:
        try:
            xs = x_shape[i]
        except IndexError:
            raise ValueError("""
                Reduction index of {} on x exceeds {}, the ndim of x
                """.format(i, len(x_shape)).strip())
        try:
            ys = y_shape[j]
        except IndexError:
            raise ValueError("""
                Reduction index of {} on y exceeds {}, the ndim of y
                """.format(j, len(y_shape)).strip())

        if xs != ys:
            raise ValueError("""
                Cannot reduce x and y due to different axis lengths.   x has shape
                {} in axis {} but y has shape {} in axis {}
                """.format(xs, i, ys, j).strip())

    if reduce_axes:
        x_axes = [i for (i, _) in reduce_axes]
        y_axes = [i for (_, i) in reduce_axes]
        if len(x_axes) != len(set(x_axes)):
            raise ValueError("""
                Reduce axes for x cannot have duplicate entries, but have x axes {}
                """.format(x_axes).strip())
        if len(y_axes) != len(set(y_axes)):
            raise ValueError("""
                Reduce axes for y cannot have duplicate entries, but have y axes {}
                """.format(x_axes).strip())

    if (elementwise_axes is None) or (elementwise_axes == []):
        elementwise_axes = []
    elif isinstance(elementwise_axes, int):
        elementwise_axes = [(elementwise_axes, elementwise_axes)]
    elif isinstance(elementwise_axes, list) and isinstance(elementwise_axes[0], int):
        elementwise_axes = [(j, j) for j in elementwise_axes]

    for (i, j) in elementwise_axes:
        try:
            xs = x_shape[i]
        except IndexError:
            raise ValueError("""
                Elementwise index of {} on x exceeds {}, the ndim of x
                """.format(i, len(x_shape)).strip())
        try:
            ys = y_shape[j]
        except IndexError:
            raise ValueError("""
                Elementwise index of {} on y exceeds {}, the ndim of y
                """.format(j, len(y_shape)).strip())

        if xs != ys:
            raise ValueError("""
                Cannot combine x and y elementwise due to different axis lengths.
                x has shape {} in axis {} but y has shape {} in axis {}
                """.format(xs, i, ys, j).strip())

    if elementwise_axes:
        x_axes = [i for (i, _) in elementwise_axes]
        y_axes = [i for (_, i) in elementwise_axes]
        if len(x_axes) != len(set(x_axes)):
            raise ValueError("""
                Elementwise axes for x cannot have duplicate entries, but have x axes {}
                """.format(x_axes).strip())
        if len(y_axes) != len(set(y_axes)):
            raise ValueError("""
                Elementwise axes for y cannot have duplicate entries, but have y axes {}
                """.format(x_axes).strip())

    if elementwise_axes + reduce_axes:
        x_elem = [i for (i, _) in elementwise_axes]
        y_elem = [i for (_, i) in elementwise_axes]
        x_reduce = [i for (i, _) in reduce_axes]
        y_reduce = [i for (_, i) in reduce_axes]
        x_axes = x_elem + x_reduce
        y_axes = y_elem + y_reduce

        if len(x_axes) != len(set(x_axes)):
            raise ValueError("""
                Elementwise and reduce axes for x cannot share entries, but have
                elementwise {} and reduce {} for x
                """.format(x_elem, x_reduce).strip())
        if len(y_axes) != len(set(y_axes)):
            raise ValueError("""
                Elementwise and reduce axes for y cannot share entries, but have
                elementwise {} and reduce {} for y
                """.format(y_elem, y_reduce).strip())

    return reduce_axes, elementwise_axes


