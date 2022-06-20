import tensorflow as tf

metric = tf.keras.losses.MeanSquaredError()
K = tf.keras.backend
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Activation = tf.keras.layers.Activation
Lambda = tf.keras.layers.Lambda
Add = tf.keras.layers.Add
LayerNorm = tf.keras.layers.LayerNormalization


class TFT(tf.keras.models.Model):
  def __init__(self,
               out_steps,
               identifier_feature_name,
               identifier_feature_count,
               label_feature_name,
               categorical_features=None,
               numerical_feature_names=None,
               category_counts=None,
               hidden_size=256,
               hidden_layer_size=3,
               features=1,
               **kwargs):
    super().__init__(**kwargs)
    self.out_steps = out_steps
    self.identifier_feature_name = identifier_feature_name
    self.identifier_feature_count = identifier_feature_count
    self.label_feature_name = label_feature_name
    self.hidden_size = hidden_size
    self.hidden_layer_size = hidden_layer_size
    self.features = features
    self.categorical_features = categorical_features
    self.category_counts = category_counts
    self.numerical_feature_names = numerical_feature_names
    self._static_input_loc = True if self.numerical_feature_names is not None and len(self.numerical_feature_names) > 0 else False

    self.numerical_inputs, self.obs_inputs = self.get_tft_embeddings()

    self._identifier = tf.keras.layers.Embedding(
      input_dim=self.identifier_feature_count,
      output_dim=self.hidden_layer_size
    )

    self.calls = tf.keras.layers.Dense(
      2, kernel_initializer=tf.initializers.zeros())

    # self.conv1d = tf.keras.layers.Conv1D(32, 8)

    self.dense_variable_selection = tf.keras.layers.Dense(
      3, kernel_initializer=tf.initializers.zeros())

    self.lstm_layer1 = tf.keras.layers.LSTM(128, return_sequences=True)
    self.lstm_layer2 = tf.keras.layers.LSTM(128, return_sequences=True)
    # self.add_and_norm = AddAndNorm()
    self.lstm_layer3 = tf.keras.layers.LSTM(64, return_sequences=False)
    self.gated_residual_network = GatedResidualNetwork(self.hidden_layer_size)
    self.multi_head = InterpretableMultiHeadAttention(
      n_head=16, d_model=self.hidden_layer_size, dropout=0.1)
    # Shape => [batch, out_steps*features].
    self.gated_residual_network_2 = GatedResidualNetwork(self.hidden_layer_size)
    self.output_layer = tf.keras.layers.Dense(self.out_steps, kernel_initializer=tf.initializers.zeros())
    self.conv = tf.keras.layers.Conv1D(2, 3)

    self.output_layer2 = tf.keras.layers.Dense(self.out_steps, kernel_initializer=tf.initializers.zeros())

    self.add_and_norm = AddAndNorm()

  def get_tft_embeddings(self):
    """Transforms raw inputs to embeddings.
    Applies linear transformation onto continuous variables and uses embeddings
    for categorical variables.
    Args:
      all_inputs: Inputs to transform
    Returns:
      Tensors for transformed inputs.
    """

    time_steps = self.out_steps

    # Sanity checks
    """
    for i in self._known_regular_input_idx:
      if i in self._input_obs_loc:
        raise ValueError('Observation cannot be known a priori!')
    for i in self._input_obs_loc:
      if i in self._static_input_loc:
        raise ValueError('Observation cannot be static!')
    """

    num_categorical_variables = len(self.category_counts)
    num_regular_variables = (len(self.numerical_feature_names)
      if self.numerical_feature_names is not None
         and len(self.numerical_feature_names) > 0 else None)

    embedding_sizes = [
      self.hidden_layer_size for i, size in enumerate(self.category_counts)
    ]

    embeddings = []
    for i in range(num_categorical_variables):
      embedding = tf.keras.layers.Embedding(
          self.category_counts[i],
          embedding_sizes[i],
          input_length=time_steps,
          dtype=tf.float32)
      embeddings.append(embedding)

    """
    # Static inputs
    if self._static_input_loc:
      static_inputs = [tf.keras.layers.Dense(self.hidden_layer_size) for i in range(num_regular_variables)
                        if i in self._static_input_loc]
    else:
      static_inputs = None

    def convert_real_to_embedding():
      return tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(self.hidden_layer_size))
    """

    # Targets
    if num_regular_variables:
      numerical_inputs = [tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.hidden_layer_size))
        for i in range(num_regular_variables)]
    else:
      numerical_inputs = None
      

    return numerical_inputs, embeddings


  @tf.function
  def train_step(self, data):
    if len(data) == 3:
      x, y_true, sample_weight = data
    else:
      x, y_true = data
    with tf.GradientTape() as tape:
      y_pred = self(x, is_training=True)
      loss = self.compiled_loss(y_true, y_pred)
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    self.compiled_metrics.update_state(y_true, y_pred)

    # Return a dict mapping metric names to current value.
    # Note that it will include the loss (tracked in self.metrics).
    return {m.name: m.result() for m in self.metrics}

  @tf.function
  def test_step(self, data):
    x, y_true = data
    y_pred = self(x, is_training=False)
    loss = self.compiled_loss(y_true=y_true, y_pred=y_pred)
    self.compiled_metrics.update_state(y_true, y_pred)

    # Return a dict mapping metric names to current value.
    # Note that it will include the loss (tracked in self.metrics).
    return {m.name: m.result() for m in self.metrics}


  def call(self, inputs, is_training=True):
    identifier = self._identifier(inputs[self.identifier_feature_name])
    input = ([layer(inputs[self.categorical_features[i]])
             for i, layer in enumerate(self.obs_inputs)]
            + [identifier]
            + [layer(tf.reshape(
                tensor=inputs[self.numerical_feature_names[i]],
                shape=[-1, self.out_steps, 1]))
               for i, layer in enumerate(self.numerical_inputs)])

    concatenated_input = tf.keras.layers.Concatenate(axis=1)(input)

    # output = self.dense_variable_selection(concatenated_input)

    output = self.lstm_layer1(concatenated_input)
    output = self.lstm_layer2(output)
    output = self.lstm_layer3(output)

    # output = self.add_and_norm([concatenated_input, output])

    print("SHAAAAAAAAAA", output.shape)

    # output = self.gated_residual_network(output)
    # output, _ = self.multi_head(output, output, output)
    # output = self.lstm_layer2(output)

    print("SHAAAAAAAAAA", output.shape)

    # output = self.gated_residual_network_2(output)
    # output = self.gated_residual_network(output)
    output, _ = self.multi_head(output, output, output)

    # print("LAAAAAAAA   SHAAAAAAAAAA", output.shape)

    # output = self.add_and_norm([concatenated_input, output])

    # print("LAAAAAAAA   SHAAAAAAAAAA", output.shape)

    # output = self.conv(output)

    output = self.output_layer(output)

    output = self.output_layer2(output)


    print("LAAAAAAAaLAAAAAAAA   SHAAAAAAAAAA", output.shape)
    return output


class GatedResidualNetwork(tf.keras.layers.Layer):
  """Applies the gated residual network (GRN) as defined in paper.
      Args:
        x: Network inputs
        hidden_layer_size: Internal state size
        output_size: Size of output layer
        dropout_rate: Dropout rate if dropout is applied
        use_time_distributed: Whether to apply network across time dimension
        additional_context: Additional context vector to use if relevant
        return_gate: Whether to return GLU gate for diagnostic purposes
      Returns:
        Tuple of tensors for: (GRN output, GLU gate)
  """


  def __init__(self,
        hidden_layer_size,
        output_size=None,
        dropout_rate=None,
        use_time_distributed=True,
        return_gate=False,
        **kwargs):

    super().__init__(**kwargs)
    self.hidden_layer_size = hidden_layer_size
    self.output_size = output_size
    self.dropout_rate = dropout_rate
    self.use_time_distributed = use_time_distributed
    self.return_gate = return_gate

    # Setup skip connection
    if output_size is None:
      output_size = hidden_layer_size
    else:
      self.linear = Dense(output_size)
      if use_time_distributed:
        self.linear = tf.keras.layers.TimeDistributed(self.linear)

    # Apply feedforward network
    self.hidden = linear_layer(
      hidden_layer_size,
      activation=None,
      use_time_distributed=use_time_distributed)
    self.additional_context = linear_layer(
        hidden_layer_size,
        activation=None,
        use_time_distributed=use_time_distributed,
        use_bias=False)
    self.hidden_layer_2 = tf.keras.layers.Activation('elu')
    self.hidden_layer_3 = linear_layer(
      hidden_layer_size,
      activation=None,
      use_time_distributed=use_time_distributed)

    self.gating_layer = GatingLayer(
      hidden_layer_size=output_size,
      dropout_rate=dropout_rate,
      use_time_distributed=use_time_distributed,
      activation=None)

    self.add_and_norm = AddAndNorm()

  def call(self, inputs, additional_context=None):

    if self.output_size is None:
        skip = inputs
    else:
        skip = self.linear(inputs)

    outputs = self.hidden(inputs)

    if additional_context is not None:
        outputs = outputs + self.additional_context(
            additional_context)

    outputs = self.hidden_layer_2(outputs)
    outputs = self.hidden_layer_3(outputs)

    gating_layer, gate = self.gating_layer(outputs)

    print(gating_layer.shape)
    print(skip.shape)

    if self.return_gate:
      return self.add_and_norm([skip, gating_layer]), gate
    else:
      return self.add_and_norm([skip, gating_layer])


# Layer utility functions.
def linear_layer(size,
                 activation=None,
                 use_time_distributed=False,
                 use_bias=True):
    """Returns simple Keras linear layer.
    Args:
      size: Output size
      activation: Activation function to apply if required
      use_time_distributed: Whether to apply layer across time
      use_bias: Whether bias should be included in layer
    """
    linear = tf.keras.layers.Dense(size, activation=activation, use_bias=use_bias)
    if use_time_distributed:
        linear = tf.keras.layers.TimeDistributed(linear)
    return linear


class AddAndNorm(tf.keras.layers.Layer):
  """Applies skip connection followed by layer normalisation.
  Args:
    inputs: List of inputs to sum for skip connection
  Returns:
    Tensor output from layer.
  """
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.add = Add()
    self.norm = LayerNorm()

  def call(self, inputs):
    outputs = self.add(inputs)
    outputs = self.norm(outputs)
    return outputs


class GatingLayer(tf.keras.layers.Layer):
  """Applies a Gated Linear Unit (GLU) to an input.
  Args:
    x: Input to gating layer
    hidden_layer_size: Dimension of GLU
    dropout_rate: Dropout rate to apply if any
    use_time_distributed: Whether to apply across time
    activation: Activation function to apply to the linear feature transform if
      necessary
  Returns:
    Tuple of tensors for: (GLU output, gate)
  """

  def __init__(
        self,
        hidden_layer_size,
        dropout_rate=None,
        use_time_distributed=True,
        activation=None,
        **kwargs):
    super().__init__(**kwargs)
    self.hidden_layer_size = hidden_layer_size
    self.dropout_rate = dropout_rate
    self.use_time_distributed = use_time_distributed
    self.activation = activation

    if self.dropout_rate is not None:
      self.drop_layer = tf.keras.layers.Dropout(dropout_rate)

    if self.use_time_distributed:
      self.activation_layer = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(hidden_layer_size, activation=activation))
      self.gated_layer = tf.keras.layers.TimeDistributed(
      tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid'))
    else:
      self.activation_layer = tf.keras.layers.Dense(
        hidden_layer_size, activation=activation)
      self.gated_layer = tf.keras.layers.Dense(
        hidden_layer_size, activation='sigmoid')

    self.multiply = tf.keras.layers.Multiply()

  def call(self, inputs):
    if self.dropout_rate is not None:
      outputs = self.drop_layer(inputs)
    else:
      outputs = inputs

    activation = self.activation_layer(outputs)
    gated_layer = self.gated_layer(outputs)
    return self.multiply([activation, gated_layer]), gated_layer


class InterpretableMultiHeadAttention(tf.keras.layers.Layer):
  """Defines interpretable multi-head attention layer.
  Attributes:
      n_head: Number of heads
      d_k: Key/query dimensionality per head
      d_v: Value dimensionality
      dropout: Dropout rate to apply
      qs_layers: List of queries across heads
      ks_layers: List of keys across heads
      vs_layers: List of values across heads
      attention: Scaled dot product attention layer
      w_o: Output weight matrix to project internal state to the original TFT
        state size
  """

  def __init__(self, n_head, d_model, dropout, **kwargs):
    """Initialises layer.
      Args:
        n_head: Number of heads
        d_model: TFT state dimensionality
        dropout: Dropout discard rate
    """
    super().__init__(**kwargs)
    self.n_head = n_head
    self.d_k = self.d_v = d_k = d_v = d_model // n_head
    self.dropout = dropout

    self.qs_layers = []
    self.ks_layers = []
    self.vs_layers = []
    self.head_dropouts = []

    # Use same value layer to facilitate interp
    vs_layer = Dense(d_v, use_bias=False)

    for _ in range(n_head):
      self.qs_layers.append(Dense(d_k, use_bias=False))
      self.ks_layers.append(Dense(d_k, use_bias=False))
      self.vs_layers.append(vs_layer)  # use same vs_layer
      self.head_dropouts.append(Dropout(self.dropout))

    self.attention = ScaledDotProductAttention()
    self.dropout_layer = Dropout(self.dropout)
    self.w_o = Dense(d_model, use_bias=False)

  def call(self, q, k, v, mask=None):
    """Applies interpretable multihead attention.
    Using T to denote the number of time steps fed into the transformer.
    Args:
        q: Query tensor of shape=(?, T, d_model)
        k: Key of shape=(?, T, d_model)
        v: Values of shape=(?, T, d_model)
        mask: Masking if required with shape=(?, T, T)
    Returns:
      Tuple of (layer outputs, attention weights)
    """
    n_head = self.n_head

    heads = []
    attns = []
    for i in range(n_head):
      qs = self.qs_layers[i](q)
      ks = self.ks_layers[i](k)
      vs = self.vs_layers[i](v)
      head, attn = self.attention(qs, ks, vs, mask)

      head_dropout = self.head_dropouts[i](head)
      heads.append(head_dropout)
      attns.append(attn)
    head = K.stack(heads) if n_head > 1 else heads[0]
    attn = K.stack(attns)

    outputs = K.mean(head, axis=0) if n_head > 1 else head
    outputs = self.w_o(outputs)
    outputs = self.dropout_layer(outputs)  # output dropout

    return outputs, attn


class ScaledDotProductAttention(tf.keras.layers.Layer):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.
  Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
        to (..., seq_len_q, seq_len_k). Defaults to None.
  Returns:
      output, attention_weights
  """

  def call(self, q, k, v, mask):
    """This is where the layer's logic lives."""
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
      scaled_attention_logits += mask * -1e9

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    # (..., seq_len_q, seq_len_k)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

