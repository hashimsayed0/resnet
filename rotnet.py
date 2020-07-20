import yaml
import os
import sys
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.client import device_lib
from resnet import ResNet
from data import Data
import shutil

class RotNet(object):
    def __init__(self, sess, args):
        print("[INFO] Reading configuration file")
        self.config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

        self.sess = sess
        self.data_dir = args.data_dir
        self.model_dir = "./checkpoints/"
        self.classes = ["0", "90", "180", "270"]
        self.model_number = args.model_number
        self.model = ResNet()

        self._populate_model_hyperparameters()
        self.data_obj = Data(self.data_dir,
                            batch_size=self.batch_size,
                            height=self.height,
                            width=self.width
                            )
        self.build_base_graph()

        if args.train:
            #If we are training, then we want to run the optimizer
            self.build_train_graph()

        #List the compute available on the device that this script is being run on.
        print(device_lib.list_local_devices())

        #This collects the add_summary operations that you defined in the graph. You should be saving your metrics to self.summary
        self.summary = tf.compat.v1.summary.merge_all()

    def _populate_model_hyperparameters(self):
        """
        This is a helper function for populating the hyperparameters from the yaml file
        """
        self.batch_size = self.config["batch_size"]
        self.weight_decay = self.config["weight_decay"]
        self.momentum = self.config["momentum"]
        self.learning_rate = self.config["learning_rate"]
        self.height = self.config["image_height"]
        self.width = self.config["image_width"]
        self.num_epochs = self.config["num_epochs"]

    def build_base_graph(self):
        #Initialize your dataloader here using tf.data by calling "get_rot_data_iterator"
        self.x_input = tf.compat.v1.placeholder(tf.float32, [None, 32, 32, 3])
        self.y_input = tf.compat.v1.placeholder(tf.float32, [None, 4])

        self.iterator = self.data_obj.get_rot_data_iterator(self.x_input, self.y_input, self.batch_size)
        images, labels = self.iterator.get_next()

        logits = self.model.forward(images)

        # Calculate the loss and accuracy from your output logits.
        # Add your accuracy metrics and loss to the tensorboard summary using tf.summary
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
        self.loss = tf.reduce_mean(entropy)

        Y_pred = tf.nn.softmax(logits)
        y_pred_cls = tf.argmax(Y_pred, axis=1)
        y_cls = tf.argmax(labels, axis=1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred_cls, y_cls), tf.float32))

        tf.compat.v1.summary.scalar('loss', tensor=self.loss)
        tf.compat.v1.summary.scalar('accuracy', tensor=self.accuracy)
        self.summary_op = tf.compat.v1.summary.merge_all()
        #END OF FUNCTION

    def build_train_graph(self):
        #Create an optimizer that minimizes the loss function that you defined in the above function
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        #This will restore a model @ the latest epoch if you have already started training
        #If it cannot find a checkpoint, it will set the starting epoch to zero
        if os.path.exists("./checkpoints/model{0}".format(self.model_number)):
            self.start_epoch = self.restore_from_checkpoint()
        else:
            self.start_epoch = 0

        #Creates a writer for Tensorboard
        self.train_writer = tf.compat.v1.summary.FileWriter("./logs/train/" + str(self.model_number), self.sess.graph)
        self.saver = tf.compat.v1.train.Saver()

    def train(self):
        # Initialize your graph variables
        self.sess.run([tf.compat.v1.global_variables_initializer()])    

        #Implement and call the get_training_data function to get the data from disk
        x, y = self.data_obj.get_training_data()
        x, y = self.data_obj.preprocess(x)

        #TSplit the data into a training and validation set: see sklearn train_test_split
        X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.33)
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=4)
        y_val = tf.keras.utils.to_categorical(y_val, num_classes=4)
        print("X_train shape:")
        print(X_train.shape)
        print("y_train shape:")
        print(y_train.shape)
        print("X_val shape:")
        print(X_val.shape)
        print("y_val shape:")
        print(y_val.shape)

        num_batches = X_train.shape[0] / self.batch_size
        step = 1
        # Implement the training and validation loop and checkpoint your file at each epoch
        print("[INFO] Starting Training...")
        for epoch in range(self.start_epoch, self.num_epochs):
            self.sess.run([self.iterator.initializer], feed_dict={self.x_input: X_train, self.y_input: y_train})
            for batch in range(int(num_batches)):
                self._update_learning_rate(epoch)
                o, loss, accuracy = self.sess.run([self.optimizer, self.loss, self.accuracy])
                self.train_writer.add_summary(tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag="loss", simple_value=loss)]))
                self.train_writer.add_summary(tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag="accuracy", simple_value=accuracy)]))

                print("Epoch: {0}, Batch: {1} ==> Accuracy: {2}, Loss: {3}".format(epoch, batch, accuracy, loss))
                step += 1
            # Calculate validation accuracy and loss
            self.sess.run(self.iterator.initializer, feed_dict={self.x_input: X_val, self.y_input: y_val})
            print("Epoch: {0}, Validation ==> Accuracy: {1}, Loss: {2}".format(epoch, accuracy, loss))
            self.save_checkpoint(step, epoch)

        #Evaluate your data on the test set after training
        images, labels = self.data_obj.get_test_data()
        X_test, y_test = self.data_obj.preprocess(images)
        y_test = tf.keras.utils.to_categorical(y_test)
        self.sess.run([self.iterator.initializer], feed_dict={self.x_input: X_test, self.y_input: y_test})
        o, loss, accuracy = self.sess.run([self.optimizer, self.loss, self.accuracy])
        print("Test accuracy:" + str(accuracy))

    def predict(self, image_path):
        self.restore_from_checkpoint()
        image = cv2.imread(image_path)
        image = self.data_obj.convert_images(image)
        logits = self.model.forward(image)
        Y_pred = tf.nn.softmax(logits)
        pred_class = tf.argmax(Y_pred, axis=1)
        return str(pred_class * 90)

    def restore_from_checkpoint(self):
        self.saver = tf.compat.v1.train.import_meta_graph('my-model-{0}.meta'.format(self.model_number))
        checkpoint = tf.compat.v1.train.latest_checkpoint('./')
        self.saver.restore(self.sess, checkpoint)        
        return 0

    def save_checkpoint(self, global_step, epoch):
        if epoch == 0:
            self.path = self.saver.save(self.sess, 'my-model', global_step=global_step, write_meta_graph=True)
        else:
            self.path = self.saver.save(self.sess, 'my-model', global_step=global_step, write_meta_graph=False)
        dir = "./checkpoints/model{0}".format(self.model_number)
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)
        return

    def _update_learning_rate(self, epoch):
        #In the paper the learning rate is updated after certain epochs to slow down learning.
        if epoch == 80 or epoch == 60 or epoch == 30:
            self.learning_rate = self.learning_rate * 0.2
