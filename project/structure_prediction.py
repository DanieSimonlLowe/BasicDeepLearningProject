import os
# suppress silly log messages from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import structure_prediction_utils as utils
from tensorflow import keras
from keras import losses
from Model import Model
from Grapher import save_loss


print(tf.config.list_physical_devices('GPU'))

batch_size = 29
#50 #1024 # small because of 

def get_n_records(batch):
    return batch['primary_onehot'].shape[0]
def get_input_output_masks(batch):
    inputs = {'primary_onehot':batch['primary_onehot']}
    outputs = batch['true_distances']
    masks = batch['distance_mask']

    return inputs, outputs, masks
def train(model, train_dataset, validate_dataset=None, train_loss=utils.mse_loss):
    '''
    Trains the model
    '''

    avg_mse_loss = 0.
    avg_loss = 0
    total_loss = 0
    total_count = 0

    def print_loss():
        if validate_dataset is not None:
            validate_loss = 0.

            validate_batches = 0.
            for batch in validate_dataset:
                validate_inputs, validate_outputs, validate_masks = get_input_output_masks(batch)
                with tf.device('/GPU:0'):
                    validate_inputs = {k: tf.convert_to_tensor(v) for k, v in validate_inputs.items()}
                    validate_outputs = tf.convert_to_tensor(validate_outputs)
                    validate_masks = tf.convert_to_tensor(validate_masks)

                context = validate_inputs['primary_onehot']
                validate_preds = model(context)

                validate_loss += tf.reduce_sum(utils.mse_loss(validate_preds, validate_outputs, validate_masks)) / get_n_records(batch)
                validate_batches += 1
            validate_loss /= validate_batches

            utils.display_two_structures(validate_preds[0],validate_outputs[0],validate_masks[0])
        else:
            validate_loss = float('NaN')
        print(
            f'train loss {avg_loss:.3f} train mse loss {avg_mse_loss:.3f} validate mse loss {validate_loss:.3f}')
        
        
    first = True
    counter = 0


    for batch in train_dataset:
        inputs, labels, masks = get_input_output_masks(batch)

        with tf.device('/GPU:0'):
            inputs = {k: tf.convert_to_tensor(v) for k, v in inputs.items()}
            lables = tf.convert_to_tensor(labels)
            masks = tf.convert_to_tensor(masks)
            
        
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            outputs = model(inputs)

            l = train_loss(outputs, labels, masks)
            batch_loss = tf.reduce_sum(l)
            gradients = tape.gradient(batch_loss, model.trainable_weights)
            #avg_loss = batch_loss / get_n_records(batch)
            total_count += get_n_records(batch)
            total_loss += batch_loss
            avg_loss = total_loss / total_count
            avg_mse_loss = tf.reduce_sum(utils.mse_loss(outputs, labels, masks)) / get_n_records(batch)
            print(counter, avg_loss)
            counter += get_n_records(batch)
            

        model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        
            

    print('epoch', avg_loss)
    #print_loss()
    

    if first:
        print(model.summary())
        first = False
    
    return avg_loss

def test(model, test_records, viz=False):
    total_loss = 0
    total_count = 0
    for batch in test_records:
        test_inputs, test_outputs, test_masks = get_input_output_masks(batch)

        test_preds = model(test_inputs)
        #test_loss = tf.reduce_sum(utils.mse_loss(test_preds, test_outputs, test_masks)) / get_n_records(batch)
        total_loss += tf.reduce_sum(utils.mse_loss(test_preds, test_outputs, test_masks))
        total_count += get_n_records(batch)
        print(f'test mse loss {total_loss/total_count:.3f}')
    return total_loss/total_count

    # if viz:
    #     print(model.summary())
    #     r = random.randint(0, test_preds.shape[0])
    #     utils.display_two_structures(test_preds[r], test_outputs[r], test_masks[r])

def main(data_folder,save_num):
    training_records = utils.load_preprocessed_data(data_folder, 'training.tfr', batch_size)
    validate_records = utils.load_preprocessed_data(data_folder, 'validation.tfr', batch_size)
    test_records = utils.load_preprocessed_data(data_folder, 'testing.tfr', batch_size)


    model = Model()
    model.optimizer = keras.optimizers.Adam(learning_rate=1e-2)
    model.loss = losses.mean_absolute_error
    model.batch_size = batch_size
    epochs = 25
    # Iterate over epochs.

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="models_1/",
        save_weights_only=True,
        monitor='val_loss',  # You can also track your custom metric
        mode='min',
        save_best_only=False
        )
    model_checkpoint_callback.set_model(model)

    for epoch in range(epochs):
        epoch_training_records = training_records#.shuffle(buffer_size=256).batch(model.batch_size, drop_remainder=False)
        print("Start of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        train_loss = train(model, epoch_training_records, validate_records)
        
        loss = test(model, test_records, True)
        save_loss(f'model_4_{save_num}.csv',epoch,train_loss,loss)
        #model.save(data_folder + f'/model_{epoch}')
        model_checkpoint_callback.on_epoch_end(epoch, logs={'val_loss':loss})
        print('epoch',epoch)

    test(model, test_records, True)

    #model.save(data_folder + '/model_end')


if __name__ == '__main__':
    #local_home = os.path.expanduser("~")  # on my system this is /Users/jat171
    #data_folder = local_home + '/Desktop/cosc440/project/'
    for i in range(3,5):
        try:
            main('',i)
        except:
            continue