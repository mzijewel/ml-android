package dev.jewel.machinelearning

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Base64
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import dev.jewel.machinelearning.ui.theme.MachineLearningTheme
import java.io.ByteArrayOutputStream


// https://www.youtube.com/watch?v=igJgXeV82b4
// https://chaquo.com/chaquopy/doc/current/android.html

class MainActivity : ComponentActivity() {
    lateinit var module: PyObject
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // load python
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }

        // load python module
        val python = Python.getInstance()
        module = python.getModule("utils") // utils.py


        val info = getInfo()
        val grayBitmap = getGrayBitmap()


        setContent {
            MachineLearningTheme {
                // A surface container using the 'background' color from the theme
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    Column {
                        Text(text = info)
                        Image(bitmap = grayBitmap.asImageBitmap(), contentDescription = "")
                    }

                }
            }
        }
    }

    fun getGrayBitmap(): Bitmap {
        val bitmap = BitmapFactory.decodeStream(assets.open("cat.jpg"))
        // Convert bitmap to byte array
        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
        val byteArray = stream.toByteArray()
        val ps = module["process_bitmap"]?.call(byteArray).toString()
        val barray = Base64.decode(ps, Base64.DEFAULT)
        return BitmapFactory.decodeByteArray(barray, 0, barray.size)
    }

    fun getInfo(): String {
        val method = module["get_info"] // method name
        val result = method?.call()
        return result.toString()
    }
}
