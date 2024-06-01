package dev.jewel.machinelearning

import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.ViewModelProvider
import coil.compose.AsyncImage
import dev.jewel.machinelearning.ui.theme.MachineLearningTheme


// https://www.youtube.com/watch?v=igJgXeV82b4
// https://chaquo.com/chaquopy/doc/current/android.html

class MainActivity : ComponentActivity() {
    private lateinit var mainVm: MainVm
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        mainVm = ViewModelProvider(this)[MainVm::class.java]
        mainVm.loadModel(this)


        setContent {
            MachineLearningTheme {
                // A surface container using the 'background' color from the theme
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    var selectedImageUri by remember {
                        mutableStateOf<Uri?>(null)
                    }

                    val imgInfo by mainVm.info.collectAsState()
                    val grayImgPath by mainVm.grayImgPath.collectAsState()
                    val nobgImgPath by mainVm.nobgImgPath.collectAsState()
                    val bmpd by mainVm.bmpd.collectAsState()

                    var imagePath by remember {
                        mutableStateOf("")
                    }
                    val singlePhotoPickerLauncher = rememberLauncherForActivityResult(
                        contract = ActivityResultContracts.PickVisualMedia(),
                        onResult = { uri ->
                            selectedImageUri = uri
                            uri?.let {
                                imagePath = it.toImagePath(this)
                                mainVm.processImage(imagePath)
                            }
                        }
                    )



                    Column(
                        modifier = Modifier
                            .verticalScroll(rememberScrollState())
                            .fillMaxSize()
                    ) {
                        AsyncImage(model = bmpd, contentDescription = "")
                        Row {
                            Button(onClick = {
                                singlePhotoPickerLauncher.launch(
                                    PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                                )
                            }) {
                                Text(text = "Pick Image")
                            }

                            nobgImgPath?.let {
                                Spacer(modifier = Modifier.width(8.dp))
                                Button(onClick = {
                                    grayImgPath?.let {
                                        mainVm.saveImage(it)
                                    }
                                    nobgImgPath?.let {
                                        mainVm.saveImage(it)
                                    }
                                }) {
                                    Text(text = "Save")
                                }
                            }
                        }
                        Spacer(modifier = Modifier.height(10.dp))
                        Row {
                            AsyncImage(
                                model = imagePath, contentDescription = "",
                                modifier = Modifier.height(200.dp),
                                contentScale = ContentScale.FillHeight
                            )
                            Spacer(modifier = Modifier.width(10.dp))
                            Text(
                                text = imgInfo,
                                color = Color.Red,
                                fontSize = 14.sp,
                                modifier = Modifier.padding(all = 4.dp)
                            )
                        }
                        Spacer(modifier = Modifier.height(10.dp))
                        AsyncImage(
                            model = grayImgPath,
                            contentDescription = "",
                            modifier = Modifier.height(200.dp),
                            contentScale = ContentScale.FillHeight
                        )
                        Spacer(modifier = Modifier.height(10.dp))
                        AsyncImage(
                            model = nobgImgPath, contentDescription = "",
                            modifier = Modifier.height(200.dp),
                            contentScale = ContentScale.FillHeight
                        )

                    }

                }
            }
        }

    }
}
