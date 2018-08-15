import * as tf from '@tensorflow/tfjs';

const IMAGE_SIZE = 784;
const NUM_DATASET_ELEMENTS = 65000;

export class myCanvas {
    constructor() {
    }

    async load(ImgSrc) {
        // Make a request for the MNIST sprited image.
        const img = new Image();
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const imgRequest = new Promise((resolve, reject) => {
            img.crossOrigin = '';
            img.onload = () => {
                img.width = img.naturalWidth;
                img.height = img.naturalHeight;

                const datasetBytesBuffer =
                    new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);

                const chunkSize = 5000;
                canvas.width = img.width;
                canvas.height = chunkSize;

                for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
                    const datasetBytesView = new Float32Array(
                        datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4,
                        IMAGE_SIZE * chunkSize);
                    ctx.drawImage(
                        img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width,
                        chunkSize);

                    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

                    for (let j = 0; j < imageData.data.length / 4; j++) {
                        // All channels hold an equal value since the image is grayscale, so
                        // just read the red channel.
                        datasetBytesView[j] = imageData.data[j * 4] / 255;
                    }
                }
                this.datasetImages = new Float32Array(datasetBytesBuffer);

                resolve();
            };
            img.src = ImgSrc;
        });

        //const [imgResponse] =
        await Promise.all([imgRequest]);

        this.testImages = this.datasetImages;


    }

    nextTestBatch(batchSize) {
        return this.nextBatch(batchSize, this.testImages);
    }

    nextBatch(batchSize, data) {
        const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);

        for (let i = 0; i < batchSize; i++) {
            const idx = 0;

            const image = data.slice(idx * IMAGE_SIZE, 784);
            batchImagesArray.set(image, i * IMAGE_SIZE);
        }

        const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);

        return xs;
    }
}
