from keras import layers
import keras
import tensorflow as tf

class DiffusionModel(keras.Model):
    def __init__(self, batch_size, network, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalizer = layers.Normalization()
        self.network = network
        self.image_size = 256
        self.batch_size = batch_size
        self.max_signal_rate = 0.95
        self.min_signal_rate = 0.02

        #self.build([(self.batch_size, self.image_size, self.image_size, 1), (self.batch_size, 256, 21)])
    # 32,64,96,128
    def denoise(self, noisy_images, noise_rates, signal_rates, context, training):
        pred_noises = self.network(noisy_images, noise_rates**2, context, training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = tf.cast(tf.math.acos(self.max_signal_rate), "float32")
        end_angle = tf.cast(tf.math.acos(self.min_signal_rate), "float32")

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = tf.math.cos(diffusion_angles)
        noise_rates = tf.math.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates
    
    def train_step(self, images, contexts, masks):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=True)
        noises = tf.random.normal(shape=(self.batch_size, self.image_size, self.image_size, 1))
        

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(self.batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        images = tf.reshape(images,tf.concat([tf.shape(images), [1]], axis=0))

        noisy_images = signal_rates * images + noise_rates * noises
        masks = tf.reshape(masks,noises.shape)
        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, contexts, training=True
            )
            noises = noises*masks
            pred_noises = pred_noises*masks
            noise_loss = self.loss(noises, pred_noises)  # used for training
            #image_loss = self.loss(images, pred_images)  # only used as metric

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))
        return tf.reduce_sum(noise_loss).numpy()

    def reverse_diffusion(self, context, initial_noise=None, diffusion_steps=20):
        # reverse diffusion = sampling
        if initial_noise is None:
            initial_noise = tf.random.normal(shape=(context.shape[0], self.image_size, self.image_size, 1))
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, context, training=False
            )
            # network used in eval mode

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        return pred_images