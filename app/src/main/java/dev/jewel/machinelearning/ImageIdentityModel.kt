package dev.jewel.machinelearning

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils

class ImageIdentityModel {
    // python: Cifar.py
    fun runCifar10(context: Context) {
        // Load the PyTorch model from the assets folder
        val module = Module.load(Utils.assetFilePath(context, "m_cifar.pt"))
        val classes= arrayListOf("plane", "car", "bird", "cat", "deer", "dog",
            "frog", "horse", "ship", "truck")
        val images = arrayListOf("cat.jpg", "dog.jpg", "car.jpeg","deer.jpeg","hen.jpeg")

        for (img in images) {
            // Load and preprocess the input image
            val bitmap = BitmapFactory.decodeStream(context.assets.open(img))
            val resizedBitmap =
                Bitmap.createScaledBitmap(bitmap, 32, 32, true) // filter true is important

            val tensorImage: Tensor = TensorImageUtils.bitmapToFloat32Tensor(
                resizedBitmap,
//            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
//                floatArrayOf(0.5f, 0.5f, 0.5f), // as normalized
                floatArrayOf(0f, 0f, 0f), // no mean
//            TensorImageUtils.TORCHVISION_NORM_STD_RGB
//                floatArrayOf(0.5f, 0.5f, 0.5f),
                floatArrayOf(1f, 1f,1f), // no std
            )

            // Perform inference
            val outputTensor = module.forward(IValue.from(tensorImage)).toTensor()


            // Convert output tensor to Java array
            val scores = outputTensor.dataAsFloatArray

            var maxScore = -Float.MAX_VALUE
            var maxScoreIdx = -1
            for (i in scores.indices) {
                if (scores[i] > maxScore) {
                    maxScore = scores[i]
                    maxScoreIdx = i
                }
            }

            // Apply softmax to get probabilities
            val probabilities = scores.map { Math.exp(it.toDouble()).toFloat() }
            val sum = probabilities.sum()
            val normalizedProbabilities = probabilities.map { it / sum }

            // Find the class with the highest probability
            val maxIndex = normalizedProbabilities.indices.maxByOrNull { normalizedProbabilities[it] } ?: -1
            val maxProbability = normalizedProbabilities[maxIndex]

            val accuracy =  maxScore/classes.size.toFloat()*100
            val ac= String.format("%.2f", accuracy) // accuracy is not correct yet
            Log.d("PyTorch", "$img: ${classes[maxScoreIdx]}:${maxScoreIdx}")
        }


    }

}