import tensorflow as tf
from tensorflow.keras import layers, initializers, applications


def ConvolutionBlock(
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    def apply(x):
        x = layers.Conv2D(
            num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="same",
            use_bias=use_bias,
            kernel_initializer=initializers.HeNormal(),
        )(x)
        x = layers.BatchNormalization()(x)
        return tf.nn.relu(x)

    return apply


def DilatedSpatialPyramidPooling():
    def apply(dspp_input):
        dims = dspp_input.shape
        x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
        x = ConvolutionBlock(kernel_size=1, use_bias=True)(x)
        out_pool = layers.UpSampling2D(
            size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
            interpolation="bilinear",
        )(x)

        out_1 = ConvolutionBlock(kernel_size=1, dilation_rate=1)(dspp_input)
        out_6 = ConvolutionBlock(kernel_size=3, dilation_rate=6)(dspp_input)
        out_12 = ConvolutionBlock(kernel_size=3, dilation_rate=12)(dspp_input)
        out_18 = ConvolutionBlock(kernel_size=3, dilation_rate=18)(dspp_input)

        x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
        output = ConvolutionBlock(kernel_size=1)(x)
        return output

    return apply


def build_deeplabv3_plus(
    image_size: int,
    num_classes: int,
    bakbone_type=applications.ResNet50,
    encoder_backbone_layer: str = "conv4_block6_2_relu",
    decoder_backbone_layer: str = "conv2_block3_2_relu",
):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    backbone_model = bakbone_type(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = DilatedSpatialPyramidPooling(
        backbone_model.get_layer(encoder_backbone_layer).output
    )

    encoder_input = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)

    decoder_input = ConvolutionBlock(num_filters=48, kernel_size=1)(
        backbone_model.get_layer(decoder_backbone_layer).output
    )

    x = layers.Concatenate(axis=-1)([encoder_input, decoder_input])
    x = ConvolutionBlock()(x)
    x = ConvolutionBlock()(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return keras.Model(inputs=model_input, outputs=model_output)
