package dev.jewel.machinelearning

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.SystemClock
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils

class SegmentModel {
    // python: deeplabv3.py
    fun run(context: Context) {
        // Load the PyTorch model from the assets folder
        val module = Module.load(Utils.assetFilePath(context, "deeplabv3_scripted_optimized.ptl"))
        val image = "deeplab.jpg"


        // Load and preprocess the input image
        val bitmap = BitmapFactory.decodeStream(context.assets.open(image))
        val resizedBitmap =
            Bitmap.createScaledBitmap(bitmap, 224, 224, true) // filter true is important

        val tensorImage: Tensor = TensorImageUtils.bitmapToFloat32Tensor(
            resizedBitmap,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB
        )

        val startTime = SystemClock.elapsedRealtime()
        // Perform inference
        val outTensors = module.forward(IValue.from(tensorImage)).toDictStringKey()
        val inferenceTime = SystemClock.elapsedRealtime() - startTime
        Log.d("PyTorch","time: $inferenceTime")
        val outputTensor: Tensor = outTensors["out"]!!.toTensor()

        val scores = outputTensor.dataAsFloatArray
        val width: Int = resizedBitmap.width
        val height: Int = resizedBitmap.height
        val intValues = IntArray(width * height)

        for (j in 0 until height) {
            for (k in 0 until width) {
                var maxi = 0
                var maxj = 0
                var maxk = 0
                var maxnum = -Double.MAX_VALUE
                for (i in 0 until 21) {
                    val score = scores[i * (width * height) + j * width + k]
                    if (score > maxnum) {
                        maxnum = score.toDouble()
                        maxi = i
                        maxj = j
                        maxk = k
                    }
                }
                if (maxi == 15) intValues[maxj * width + maxk] =
                    -0x10000 else if (maxi == 12) intValues[maxj * width + maxk] =
                    -0xff0100 else if (maxi == 17) intValues[maxj * width + maxk] =
                    -0xffff01 else intValues[maxj * width + maxk] = -0x1000000
            }
        }

        val bmpSegmentation = Bitmap.createScaledBitmap(resizedBitmap, width, height, true)
        val outputBitmap = bmpSegmentation.copy(bmpSegmentation.config, true)
        outputBitmap.setPixels(
            intValues,
            0,
            outputBitmap.width,
            0,
            0,
            outputBitmap.width,
            outputBitmap.height
        )
        val transferredBitmap =
            Bitmap.createScaledBitmap(outputBitmap, resizedBitmap.width, resizedBitmap.height, true)
        Log.d("PyTorch", "TensorImage")

    }

}