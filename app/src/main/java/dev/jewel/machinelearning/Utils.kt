package dev.jewel.machinelearning

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.io.IOException


object Utils {
    // Helper method to load model file from assets directory
    fun assetFilePath(context: Context, assetName: String): String? {
        val file = File(context.filesDir, assetName)
        try {
            context.assets.open(assetName).use { inStream ->
                FileOutputStream(file).use { os ->
                    val buffer = ByteArray(4 * 1024)
                    var read: Int
                    while (inStream.read(buffer).also { read = it } != -1) {
                        os.write(buffer, 0, read)
                    }
                    os.flush()
                }
            }
        } catch (e: IOException) {
            e.printStackTrace()
        }
        return file.absolutePath
    }

    fun convertToGrayscale(bitmap: Bitmap): Bitmap {
        val width = bitmap.width
        val height = bitmap.height

        val grayscaleBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        for (x in 0 until width) {
            for (y in 0 until height) {
                val pixel = bitmap.getPixel(x, y)
                val red = Color.red(pixel)
                val green = Color.green(pixel)
                val blue = Color.blue(pixel)
                val gray = (0.299 * red + 0.587 * green + 0.114 * blue).toInt()
                grayscaleBitmap.setPixel(x, y, Color.rgb(gray, gray, gray))
            }
        }

        return grayscaleBitmap
    }

    fun bitmapToInputTensor(bitmap: Bitmap, inputWidth: Int, inputHeight: Int): Tensor {
        // Resize the bitmap to the input size expected by the model
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true)

        // Convert the bitmap to a float array
        val floatValues = FloatArray(inputWidth * inputHeight)
        var pixel = 0
        for (y in 0 until inputHeight) {
            for (x in 0 until inputWidth) {
                val normalizedPixelValue = (resizedBitmap.getPixel(x, y) and 0xff) / 255.0f
                floatValues[pixel++] = normalizedPixelValue
            }
        }

        // Create a PyTorch tensor from the float array
        val inputTensor =
            Tensor.fromBlob(floatValues, longArrayOf(1, inputHeight.toLong(), inputWidth.toLong()))

        return inputTensor
    }

    fun tensorToBitmap(tensor: Tensor): Bitmap {
        // Assuming tensor shape is [1,channel, height, width]
        val height = tensor.shape()[2].toInt()
        val width = tensor.shape()[3].toInt()

        // Convert tensor to float array
        val floatValues = tensor.dataAsFloatArray

        // Create a bitmap with ARGB_8888 configuration
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        // Copy float values to bitmap pixels
        var pixel = 0
        for (y in 0 until height) {
            for (x in 0 until width) {
                val normalizedPixelValue = floatValues[pixel++] * 255
                val pixelColor =
                    (0xff shl 24) or (normalizedPixelValue.toInt() shl 16) or (normalizedPixelValue.toInt() shl 8) or normalizedPixelValue.toInt()
                bitmap.setPixel(x, y, pixelColor)
            }
        }

        return bitmap
    }
}