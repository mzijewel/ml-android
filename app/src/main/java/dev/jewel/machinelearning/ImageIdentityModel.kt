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
    // todo: predictions are not like as python, need to work on it
    fun runCifar10(context: Context) {
        // Load the PyTorch model from the assets folder
        val module = Module.load(Utils.assetFilePath(context, "m_cifar.pt"))
        val classes= arrayListOf("plane", "car", "bird", "cat", "deer", "dog",
            "frog", "horse", "ship", "truck")
        val images = arrayListOf("plane.jpg", "car.jpeg","cat.jpg",
            "deer.jpeg","dog.jpg", "hen.jpeg")

        for (img in images) {
            // Load and preprocess the input image
            val bitmap = BitmapFactory.decodeStream(context.assets.open(img))
            val resizedBitmap =
                Bitmap.createScaledBitmap(bitmap, 32, 32, true) // filter true is important

            val tensorImage: Tensor = TensorImageUtils.bitmapToFloat32Tensor(
                resizedBitmap,
//            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                floatArrayOf(0.5f, 0.5f, 0.5f), // as normalized
//                floatArrayOf(0f, 0f, 0f), // no mean
//            TensorImageUtils.TORCHVISION_NORM_STD_RGB
                floatArrayOf(0.5f, 0.5f, 0.5f),
//                floatArrayOf(1f, 1f,1f), // no std
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
            val accuracy =  maxScore/classes.size.toFloat()*100
            val ac= String.format("%.2f", accuracy)
            Log.d("PyTorch", "$img: ${classes[maxScoreIdx]}:${maxScoreIdx}")
        }


    }

}