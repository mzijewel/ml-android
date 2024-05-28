package dev.jewel.machinelearning

import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy

class ImageAnalyzer(
    private val classifier: ImageClassifierHelper,
) : ImageAnalysis.Analyzer {

    private var frameSkipCounter = 0

    override fun analyze(image: ImageProxy) {
        if (frameSkipCounter % 10 == 0) {
            val rotationDegrees = image.imageInfo.rotationDegrees
            val bitmap = image
                .toBitmap().centerCrop(320,320)

            classifier.classify(bitmap, rotationDegrees)
        }
        frameSkipCounter++

        image.close()
    }
}