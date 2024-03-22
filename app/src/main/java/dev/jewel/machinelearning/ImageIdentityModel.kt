package dev.jewel.machinelearning

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils


/* simple equation
    y=3x+2
 */
class ImageIdentityModel {
    fun run(context: Context){
        // Load the PyTorch model from the assets folder
        val module = Module.load(Utils.assetFilePath(context, "model.ptl"))

        // Load and preprocess the input image

        // Load and preprocess the input image
        val bitmap = BitmapFactory.decodeStream(context.assets.open("cat.png"))
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val tensorImage: Tensor = TensorImageUtils.bitmapToFloat32Tensor(
            resizedBitmap,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB
        )

        // Perform inference
        val outputTensor = module.forward(IValue.from(tensorImage)).toTensor()


        // Convert output tensor to Java array
        val scores = outputTensor.dataAsFloatArray

//        val classOutput=module.runMethod("get_classes")
//        val classList=classOutput.toList()
//        var classes= arrayListOf<String>()
//        for (i in classList.indices){
//            classes.add(classList[i].toStr())
//        }

        var maxScore = -Float.MAX_VALUE
        var maxScoreIdx = -1
        for (i in scores.indices) {
            if (scores[i] > maxScore) {
                maxScore = scores[i]
                maxScoreIdx = i
            }
        }

        Log.d("PyTorch","max: $maxScoreIdx")
    }
}