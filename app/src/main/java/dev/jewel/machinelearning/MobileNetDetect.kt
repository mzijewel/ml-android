package dev.jewel.machinelearning

import android.content.Context
import android.graphics.BitmapFactory
import android.util.Log
import org.pytorch.IValue
import org.pytorch.MemoryFormat
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils

class MobileNetDetect {
    fun run(context: Context){
//        val modelName="m_image.pt"
        val modelName="mnet.pt"

        val imageName="burger.jpg"
//        val imageName="hen.jpeg"
//        val imageName="cat.png"

        val module = Module.load(Utils.assetFilePath(context, modelName))
        val bitmap=BitmapFactory.decodeStream(context.assets.open(imageName))

        val inputTensor=TensorImageUtils.bitmapToFloat32Tensor(
            bitmap,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB,
            MemoryFormat.CHANNELS_LAST
        )

        val outputTensor=module.forward(IValue.from(inputTensor)).toTensor()
        val scores=outputTensor.dataAsFloatArray

        var maxScore = -Float.MAX_VALUE
        var maxScoreIdx = -1
        for (i in scores.indices) {
            if (scores[i] > maxScore) {
                maxScore = scores[i]
                maxScoreIdx = i
            }
        }
        val className = MobileNetClasses.MOBILENET_CLASSES[maxScoreIdx]

        Log.e("PyTorch","predict:$maxScoreIdx : $className")
    }
}