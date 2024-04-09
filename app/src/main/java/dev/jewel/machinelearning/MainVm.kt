package dev.jewel.machinelearning

import android.content.Context
import android.graphics.Bitmap
import android.os.Environment
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import kotlin.random.Random
import kotlin.system.measureTimeMillis

class MainVm : ViewModel() {
    private lateinit var module: PyObject
    val info = MutableStateFlow("")
    val grayImgPath = MutableStateFlow<String?>(null)
    val nobgImgPath = MutableStateFlow<String?>(null)

    init {
        Log.e("TEST", "init vm")
    }

    fun loadModel(context: Context) {
        viewModelScope.launch(Dispatchers.IO) {
            // need to call within activity or application
            if (!Python.isStarted()) {
                Python.start(AndroidPlatform(context))
            }
            val python = Python.getInstance()
            module = python.getModule("utils") // utils.py
        }

    }

    fun processImage(path: String) {
        viewModelScope.launch(Dispatchers.IO) {
            nobgImgPath.emit(null)
            grayImgPath.emit(null)
            info.emit("Image processing ...\nimage shape: ${getImageInfo(path)}")
            val time1 = measureTimeMillis {
                grayImgPath.emit(grayBitmap(path))
            }
            info.update { current ->
                "$current\nGray time: $time1 ms"
            }
            val time2 = measureTimeMillis {
                nobgImgPath.emit(removeBg(path))
            }
            info.update { currentValue ->
                "$currentValue\nRemove bg time: $time2 ms\nprocess completed"
            }
        }
    }

    private fun getImageInfo(path: String): String {
        val info = module.callAttr("image_info", path)
        return info.toString()
    }

    private fun removeBg(path: String): String {
        return module["remove_bg"]?.call(path).toString()
    }

    private fun grayBitmap(path: String): String {
        return module["gray_image"]?.call(path).toString()
    }

    private fun getInfo(): String {
        val method = module["get_info"] // method name
        val result = method?.call()
        return result.toString()
    }

    fun saveImage(path: String) {
        val number = Random.nextInt()
        val ext = path.lastIndexOf(".").let {
            path.substring(it + 1)
        }
        val fileName = "IMG_$number.$ext"

        // Get the directory for saving files
        val storageDir = File(
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),
            "Test" // Change this to your desired directory name
        )

        // Create the directory if it doesn't exist
        if (!storageDir.exists()) {
            storageDir.mkdirs()
        }

        val destPath = "$storageDir${File.separator}$fileName"
        copyFile(path, destPath)
        info.update {
            "$it\nSaved into $destPath"
        }
    }

    private fun copyFile(sourcePath: String, destinationPath: String) {
        val sourceFile = File(sourcePath)
        val destinationFile = File(destinationPath)

        FileInputStream(sourceFile).use { inputStream ->
            FileOutputStream(destinationFile).use { outputStream ->
                inputStream.copyTo(outputStream)
            }
        }
    }

    private fun saveBitmap(bitmap: Bitmap): String? {

        val number = Random.nextInt()
        val fileName = "IMG_$number.jpg"

        // Get the directory for saving files
        val storageDir = File(
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),
            "Test" // Change this to your desired directory name
        )

        // Create the directory if it doesn't exist
        if (!storageDir.exists()) {
            storageDir.mkdirs()
        }

        // Create the file object
        val file = File(storageDir, fileName)

        try {
            // Create a file output stream
            val fos = FileOutputStream(file)

            // Compress the bitmap and save it to the file
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fos)

            // Close the file output stream
            fos.close()

            // Return the absolute path of the saved file
            return file.absolutePath
        } catch (e: IOException) {
            e.printStackTrace()
        }

        return null
    }
}