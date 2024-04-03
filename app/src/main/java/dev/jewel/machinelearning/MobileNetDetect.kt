package dev.jewel.machinelearning

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import org.pytorch.IValue
import org.pytorch.MemoryFormat
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils

class MobileNetDetect {
    // python: MobileNet.py
    fun run(context: Context){
//        val modelName="m_image.pt"
        val modelName="mnet.pt"

        val classes=MobileNetClasses.MOBILENET_CLASSES

        val module = Module.load(Utils.assetFilePath(context, modelName))
        val images= listOf("burger.jpg","car.jpeg","cat.jpg","plane.jpg","dog.jpg")
        for (imageName in images) {
            val bitmap = BitmapFactory.decodeStream(context.assets.open(imageName))
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, false)
            val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                resizedBitmap,
//                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                floatArrayOf(0.5f, 0.5f, 0.5f), // as normalized
//                TensorImageUtils.TORCHVISION_NORM_STD_RGB,
                floatArrayOf(0.5f, 0.5f, 0.5f), // as normalized
                MemoryFormat.CHANNELS_LAST
            )

            val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()
            val scores = outputTensor.dataAsFloatArray

            var maxScore = -Float.MAX_VALUE
            var maxScoreIdx = -1
            for (i in scores.indices) {
                if (scores[i] > maxScore) {
                    maxScore = scores[i]
                    maxScoreIdx = i
                }
            }
            val className = classes[maxScoreIdx]

            // this accuracy is not correct yet
            val accuracy = maxScore/classes.size.toFloat()*100
            val accr=String.format("%.2f", accuracy)

            Log.e("PyTorch", "$imageName: $className ($maxScoreIdx)")
        }
    }
}