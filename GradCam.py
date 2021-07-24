import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

class draw(object):
    def __init__(self,model_path,image_path,draw_image):
        self.model_path = model_path
        self.image_path = image_path
        self.load_model()
        self.predict()
        self.draw_grad_cam(draw_image)


    def load_model(self):
        self.model = tf.keras.models.load_model(self.model_path)

    def load_predict_image(self, image_path):
        img_data = list()
        with open (image_path) as f:
            line = f.readline().replace("\n","")
            while line:
                img = plt.imread(line)
                img = cv2.resize(img, (64, 64), cv2.INTER_AREA)
                img_data.append(img)
                line = f.readline().replace("\n","")
        img_data = np.array(img_data).astype('float32')
        img_data = img_data / 255.0
        return img_data

    def img_process(self, draw_image):
        img = plt.imread(draw_image)
        img = cv2.resize(img, (64, 64), cv2.INTER_AREA)
        img = np.array(img).astype('float32')
        img = img / 255.0
        img = img[np.newaxis,...]
        return img

    def predict(self):
        preds_prob = self.model.predict(self.load_predict_image(self.image_path))
        print('\n',preds_prob)
        self.preds_class = np.argmax(preds_prob, axis=1)
        print(self.preds_class)
        
    def draw_grad_cam(self,draw_image):
        image = self.img_process(draw_image)
        original_img = cv2.imread(draw_image)
        heatmaps = list()
        layers = [t.name for t in self.model.layers if 'conv' in t.name]
        for layer in layers:
            last_conv_layer = self.model.get_layer(layer)
            grad_model = tf.keras.models.Model([self.model.inputs], [last_conv_layer.output, self.model.output])
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(image)
                loss = predictions[:,self.preds_class[0]]
            grads = tape.gradient(loss, conv_outputs)
            sum_grads = tf.reduce_sum(grads, axis=(0,1,2))
            convs_list = list()
            for i in range(sum_grads.shape[0]):
                convs_list.append(conv_outputs[0,:,:,i] * sum_grads[i])
            heatmap = tf.reduce_sum(convs_list, axis=0)
            heatmap = tf.maximum(heatmap.numpy(), 0)

            if np.max(heatmap) != 0:
                heatmap /= np.max(heatmap)
            heatmap = cv2.resize(heatmap.numpy(), (64, 64))
            heatmap = np.uint8(255 * heatmap)
            heatmaps.append(heatmap)
        
        heatmaps = np.array(heatmaps)
        heatmaps_max = np.max(heatmaps, axis=0)
        heatmaps_mean = np.mean(heatmaps, axis=0)


        plt.figure(figsize=(20,15))

        plt.subplot(131)
        plt.title('original img')
        plt.imshow(image[0][...,::-1])

        plt.subplot(132)
        plt.title('heatmaps_max')
        plt.imshow(image[0][...,::-1], alpha = 0.6)
        plt.imshow(heatmaps_max, cmap='jet', alpha=0.4)

        plt.subplot(133)
        plt.title('heatmaps_mean')
        plt.imshow(image[0][...,::-1], alpha = 0.6)
        plt.imshow(heatmaps_mean, cmap='jet', alpha=0.4)
        plt.savefig("Gram_cam.png")
        plt.pause(0.05)
